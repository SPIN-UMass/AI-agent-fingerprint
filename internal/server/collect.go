package server

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// CollectEntry is the top-level payload sent by logger.js at page unload.
type CollectEntry struct {
	Session    string            `json:"session"`
	Page       string            `json:"page"`
	UserAgent  string            `json:"userAgent"`
	EventCount int               `json:"eventCount"`
	Batch      []json.RawMessage `json:"batch"`
	ReceivedAt time.Time         `json:"receivedAt"`
}

// collectWriter serialises writes to the interaction log file.
type collectWriter struct {
	mu  sync.Mutex
	dir string
}

func newCollectWriter(logDir string) *collectWriter {
	return &collectWriter{dir: logDir}
}

// Write appends one CollectEntry as a JSONL line to interactions-YYYY-MM-DD.jsonl
func (cw *collectWriter) Write(entry *CollectEntry) error {
	cw.mu.Lock()
	defer cw.mu.Unlock()

	date := entry.ReceivedAt.UTC().Format("2006-01-02")
	path := filepath.Join(cw.dir, "interactions-"+date+".jsonl")

	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	line, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	_, err = f.Write(line)
	return err
}

// CollectHandler returns an http.Handler for POST /collect.
// It accepts the sendBeacon payload from logger.js and persists it to disk.
func CollectHandler(logDir string) http.Handler {
	cw := newCollectWriter(logDir)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// sendBeacon always uses POST
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read body — sendBeacon payloads are small (<5 MB)
		body, err := io.ReadAll(io.LimitReader(r.Body, 5<<20))
		if err != nil {
			log.Printf("[collect] read error: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		var entry CollectEntry
		if err := json.Unmarshal(body, &entry); err != nil {
			log.Printf("[collect] json parse error: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		entry.ReceivedAt = time.Now().UTC()

		if err := cw.Write(&entry); err != nil {
			log.Printf("[collect] write error: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		log.Printf("[collect] session=%s page=%s events=%d", entry.Session, entry.Page, entry.EventCount)

		// 204 No Content — sendBeacon ignores the response body anyway
		w.WriteHeader(http.StatusNoContent)
	})
}