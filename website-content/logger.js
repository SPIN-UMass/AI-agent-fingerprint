/**
 * logger.js — Silent in-memory interaction logger
 *
 * Design goal: zero mid-session HTTP requests so the server sees only
 * pure, uncontaminated browser fingerprints during normal browsing.
 *
 * Strategy:
 *   - All events are buffered in memory.
 *   - Buffer is flushed via sendBeacon (fire-and-forget, no blocking) when:
 *       1. Page is closed / navigated away (beforeunload, pagehide)
 *       2. Tab goes hidden (visibilitychange → hidden)
 *       3. Buffer reaches MAX_BUFFER — flush and continue capturing
 *   - Each flush drains the buffer completely, so multiple flushes per
 *     session produce separate JSONL entries on the server, all sharing
 *     the same sessionId for reassembly.
 *   - If sendBeacon is unavailable, the buffer is silently dropped to
 *     preserve the no-mid-session-request guarantee.
 *
 * Captured events:
 *   mousemove, mousedown, mouseup, click, dblclick, contextmenu
 *   keydown, keyup
 *   scroll, resize
 *   input, change, focus, blur
 *   touchstart, touchend
 *   selectionchange
 *   page load/unload/pagehide, visibilitychange
 *   custom (window.logEvent, window.addLog)
 */

(function () {
  'use strict';

  // ── CONFIG ─────────────────────────────────────────────────────────────────
  var COLLECT_URL        = window.location.origin + '/collect';
  var MOUSEMOVE_THROTTLE = 16;    // ms — ~60fps ceiling, keeps payload sane
  var MAX_BUFFER         = 5000;  // events — flush when reached, then continue

  // ── STATE ──────────────────────────────────────────────────────────────────
  var buffer      = [];
  var flushSeq    = 0;            // increments each flush — for server reassembly
  var sessionId   = generateId();
  var pageId      = document.title || window.location.pathname.split('/').pop() || 'unknown';
  var t0          = Date.now();
  var p0          = performance.now();
  var lastMouseMs = -Infinity;

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

  // ── RECORD ─────────────────────────────────────────────────────────────────
  /** Append one event to the in-memory buffer. Flushes if buffer is full. */
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

    buffer.push(entry);

    // Mid-session overflow flush — drains buffer, then capturing continues.
    // Uses sendBeacon so it never blocks or adds a visible request mid-session.
    if (buffer.length >= MAX_BUFFER) {
      flush('buffer_full');
    }
  }

  // ── FLUSH ──────────────────────────────────────────────────────────────────
  /**
   * Drain the buffer and POST it via sendBeacon.
   * Safe to call multiple times — each call sends whatever is in the buffer
   * at that moment and resets it. All payloads share the same sessionId so
   * the server can reassemble a full session from multiple entries.
   *
   * @param {string} reason  'unload' | 'pagehide' | 'hidden' | 'buffer_full' | 'manual'
   */
  function flush(reason) {
    if (buffer.length === 0) return;
    if (!navigator.sendBeacon) return;

    flushSeq += 1;

    var payload = JSON.stringify({
      session:     sessionId,
      page:        pageId,
      userAgent:   navigator.userAgent,
      flushSeq:    flushSeq,      // 1 = first flush, 2 = second, etc.
      flushReason: reason,        // why this flush was triggered
      eventCount:  buffer.length,
      batch:       buffer.splice(0, buffer.length)  // drain — buffer is now []
    });

    navigator.sendBeacon(
      COLLECT_URL,
      new Blob([payload], { type: 'application/json' })
    );
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

  // beforeunload — fires on navigation and tab close in most browsers
  window.addEventListener('beforeunload', function () {
    record('page', { action: 'unload' });
    flush('unload');
  });

  // pagehide — most reliable cross-browser session-end signal,
  // fires even when beforeunload is skipped (bfcache, window close)
  window.addEventListener('pagehide', function () {
    record('page', { action: 'pagehide' });
    flush('pagehide');
  });

  // visibilitychange — catches tab switching, mobile backgrounding
  document.addEventListener('visibilitychange', function () {
    record('visibility', { state: document.visibilityState });
    if (document.visibilityState === 'hidden') {
      flush('hidden');
    }
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
      key:   e.key,
      code:  e.code,
      ctrl:  e.ctrlKey,
      shift: e.shiftKey,
      alt:   e.altKey,
      meta:  e.metaKey,
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
      scrollX:     window.scrollX,
      scrollY:     window.scrollY,
      elScrollTop: el.scrollTop || 0,
      target:      desc(e.target)
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
   * Mapped to a generic app_event record — no scenario HTML needs changing.
   */
  window.addLog = function (logId, eventType, detail) {
    record('app_event', { logId: logId, event: eventType, detail: detail });
  };

  window.clearLog = function (logId) {
    record('app_event', { logId: logId, event: 'clear', detail: 'log cleared' });
  };

  /** Exposed for DevTools debugging */
  window.__sessionLog = {
    getBuffer:  function () { return buffer.slice(); },
    getSeq:     function () { return flushSeq; },
    sessionId:  sessionId,
    flushNow:   function () { flush('manual'); }
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