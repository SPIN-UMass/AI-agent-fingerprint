/**
 * logger.js — Silent in-memory interaction logger
 *
 * Design goal: zero mid-session HTTP requests so the server sees only
 * pure, uncontaminated browser fingerprints during normal browsing.
 *
 * Strategy:
 *   - All events are buffered in memory (window.__sessionLog).
 *   - At page unload, one fire-and-forget sendBeacon POST is sent to
 *     /collect with the full session payload. sendBeacon does not block
 *     navigation and does not generate a visible mid-session request.
 *   - If sendBeacon is unavailable (rare), the buffer is silently dropped
 *     to preserve the no-mid-session-request guarantee.
 *
 * Captured events:
 *   mousemove, mousedown, mouseup, click, dblclick, contextmenu
 *   keydown, keyup
 *   scroll, resize
 *   input, change, focus, blur
 *   touchstart, touchend
 *   selectionchange
 *   page load/unload, visibilitychange
 *   custom (window.logEvent)
 */

(function () {
  'use strict';

  // ── CONFIG ─────────────────────────────────────────────────────────────────
  var COLLECT_URL        = window.location.origin + '/collect';
  var MOUSEMOVE_THROTTLE = 16;   // ms — ~60fps ceiling, keeps payload sane
  var MAX_BUFFER         = 5000; // hard cap — drop oldest if exceeded

  // ── STATE ──────────────────────────────────────────────────────────────────
  var buffer       = [];
  var sessionId    = generateId();
  var pageId       = document.title || window.location.pathname.split('/').pop() || 'unknown';
  var t0           = Date.now();
  var p0           = performance.now();
  var lastMouseMs  = -Infinity;

  // ── HELPERS ────────────────────────────────────────────────────────────────
  function generateId() {
    return Math.random().toString(36).slice(2, 10) +
           Math.random().toString(36).slice(2, 10);
  }

  /** Wall-clock ms with sub-ms precision */
  function nowMs() {
    return Math.round((t0 + (performance.now() - p0)) * 1000) / 1000;
  }

  function isoNow() {
    return new Date(nowMs()).toISOString();
  }

  /** Append one event to the in-memory buffer */
  function record(type, data) {
    var entry = {
      t:       isoNow(),
      ms:      nowMs(),
      session: sessionId,
      page:    pageId,
      type:    type
    };
    if (data) {
      for (var k in data) {
        if (Object.prototype.hasOwnProperty.call(data, k)) {
          entry[k] = data[k];
        }
      }
    }

    // Ring-buffer: drop oldest when cap is hit
    if (buffer.length >= MAX_BUFFER) {
      buffer.shift();
    }
    buffer.push(entry);
  }

  // ── FLUSH (unload only) ───────────────────────────────────────────────────
  /**
   * Called once at beforeunload. Uses sendBeacon so the browser completes
   * the POST even as the page is torn down, without blocking navigation.
   * This is the ONLY outbound request the logger ever makes.
   */
  var flushed = false;

  function flushOnUnload() {
    if (flushed || buffer.length === 0) return;
    if (!navigator.sendBeacon) return;

    flushed = true; // prevent double-send

    var payload = JSON.stringify({
      session:    sessionId,
      page:       pageId,
      userAgent:  navigator.userAgent,
      eventCount: buffer.length,
      batch:      buffer.splice(0, buffer.length)
    });

    navigator.sendBeacon(COLLECT_URL, new Blob([payload], { type: 'application/json' }));
  }

  // ── PAGE LIFECYCLE ────────────────────────────────────────────────────────
  window.addEventListener('load', function () {
    record('page', {
      action:   'load',
      url:      window.location.href,
      referrer: document.referrer,
      w:        window.innerWidth,
      h:        window.innerHeight
    });
  });

  window.addEventListener('beforeunload', function () {
    record('page', { action: 'unload' });
    flushOnUnload();
  });

  document.addEventListener('visibilitychange', function () {
    record('visibility', { state: document.visibilityState });
    if (document.visibilityState === 'hidden') {
      flushOnUnload();
    }
  });

  window.addEventListener('pagehide', function () {
    record('page', { action: 'pagehide' });
    flushOnUnload();
  });

  // ── MOUSE ─────────────────────────────────────────────────────────────────
  document.addEventListener('mousemove', function (e) {
    var t = performance.now();
    if (t - lastMouseMs < MOUSEMOVE_THROTTLE) return;
    lastMouseMs = t;
    record('mousemove', { x: e.clientX, y: e.clientY, px: e.pageX, py: e.pageY });
  }, { passive: true });

  document.addEventListener('mousedown', function (e) {
    record('mousedown', { x: e.clientX, y: e.clientY, button: e.button, target: desc(e.target) });
  });

  document.addEventListener('mouseup', function (e) {
    record('mouseup', { x: e.clientX, y: e.clientY, button: e.button, target: desc(e.target) });
  });

  document.addEventListener('click', function (e) {
    record('click', { x: e.clientX, y: e.clientY, button: e.button, target: desc(e.target) });
  });

  document.addEventListener('dblclick', function (e) {
    record('dblclick', { x: e.clientX, y: e.clientY, target: desc(e.target) });
  });

  document.addEventListener('contextmenu', function (e) {
    record('contextmenu', { x: e.clientX, y: e.clientY, target: desc(e.target) });
  });

  // ── KEYBOARD ──────────────────────────────────────────────────────────────
  document.addEventListener('keydown', function (e) {
    record('keydown', {
      key: e.key, code: e.code,
      ctrl: e.ctrlKey, shift: e.shiftKey, alt: e.altKey, meta: e.metaKey,
      target: desc(e.target)
    });
  });

  document.addEventListener('keyup', function (e) {
    record('keyup', { key: e.key, code: e.code, target: desc(e.target) });
  });

  // ── SCROLL ────────────────────────────────────────────────────────────────
  document.addEventListener('scroll', function (e) {
    var el = e.target === document ? document.documentElement : e.target;
    record('scroll', {
      scrollX:      window.scrollX,
      scrollY:      window.scrollY,
      elScrollTop:  el.scrollTop || 0,
      target:       desc(e.target)
    });
  }, { passive: true, capture: true });

  // ── INPUT / FORM ──────────────────────────────────────────────────────────
  document.addEventListener('input', function (e) {
    var el  = e.target;
    var val = el.type === 'password' ? '***' : (el.value || '');
    record('input', { target: desc(el), value: val.slice(0, 200) });
  }, true);

  document.addEventListener('change', function (e) {
    var el  = e.target;
    var val = el.type === 'password' ? '***' : (el.value || '');
    record('change', { target: desc(el), value: val.slice(0, 200) });
  }, true);

  document.addEventListener('focus', function (e) {
    record('focus', { target: desc(e.target) });
  }, true);

  document.addEventListener('blur', function (e) {
    record('blur', { target: desc(e.target) });
  }, true);

  // ── TOUCH ─────────────────────────────────────────────────────────────────
  document.addEventListener('touchstart', function (e) {
    var t = e.touches[0];
    if (t) record('touchstart', { x: t.clientX, y: t.clientY, count: e.touches.length });
  }, { passive: true });

  document.addEventListener('touchend', function (e) {
    var t = e.changedTouches[0];
    if (t) record('touchend', { x: t.clientX, y: t.clientY });
  }, { passive: true });

  // ── RESIZE ────────────────────────────────────────────────────────────────
  window.addEventListener('resize', function () {
    record('resize', { w: window.innerWidth, h: window.innerHeight });
  });

  // ── SELECTION ─────────────────────────────────────────────────────────────
  document.addEventListener('selectionchange', function () {
    var sel = window.getSelection();
    if (sel && sel.toString().length > 0) {
      record('selection', { text: sel.toString().slice(0, 200) });
    }
  });

  // ── PUBLIC API ────────────────────────────────────────────────────────────
  /** Called by scenario scripts to log named application events */
  window.logEvent = function (type, data) {
    record(type, data || {});
  };

  /**
   * Compatibility shim: older scenario pages call addLog(logId, event, detail).
   * We map this to a generic app_event record so no scenario code needs changing.
   */
  window.addLog = function (logId, eventType, detail) {
    record('app_event', { logId: logId, event: eventType, detail: detail });
  };

  window.clearLog = function (logId) {
    record('app_event', { logId: logId, event: 'clear', detail: 'log cleared' });
  };

  /** Internal inspection — not used in production */
  window.__sessionLog = {
    getBuffer:  function () { return buffer.slice(); },
    sessionId:  sessionId,
    flushNow:   flushOnUnload   // manual trigger for debugging
  };

  // ── TARGET DESCRIPTOR ────────────────────────────────────────────────────
  function desc(el) {
    if (!el || el === document) return 'document';
    var parts = [el.tagName ? el.tagName.toLowerCase() : '?'];
    if (el.id) parts.push('#' + el.id);
    if (el.className && typeof el.className === 'string') {
      var cls = el.className.trim().split(/\s+/).join('.');
      if (cls) parts.push('.' + cls);
    }
    var txt = (el.textContent || el.value || '').trim().slice(0, 40);
    if (txt) parts.push('[' + txt + ']');
    return parts.join('');
  }

})();