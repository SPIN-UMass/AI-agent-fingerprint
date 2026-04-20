package server

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"agent-scraper/internal/logging"
)

func logInputEvents(session, page string, batch []json.RawMessage) {
	for _, raw := range batch {
		var ev logging.BatchEvent
		if err := json.Unmarshal(raw, &ev); err != nil {
			continue
		}
		switch ev.Type {
		case "input", "change":
			log.Printf("[input] session=%s page=%s target=%s value=%q",
				session, page, ev.Target, ev.Value)
		case "app_event":
			log.Printf("[app_event] session=%s page=%s logId=%s event=%s detail=%q",
				session, page, ev.LogID, ev.Event, ev.Detail)
		}
	}
}

func CollectHandler(logger *logging.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(io.LimitReader(r.Body, 5<<20))
		if err != nil {
			log.Printf("[collect] read error: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		var entry logging.InteractionLog
		if err := json.Unmarshal(body, &entry); err != nil {
			log.Printf("[collect] json parse error: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		entry.ReceivedAt = time.Now().UTC()

		logInputEvents(entry.Session, entry.Page, entry.Batch)

		if err := logger.WriteInteraction(&entry); err != nil {
			log.Printf("[collect] write error: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		log.Printf("[collect] session=%s page=%s events=%d flush=%s",
			entry.Session, entry.Page, entry.EventCount, entry.FlushReason)

		w.WriteHeader(http.StatusNoContent)
	})
}
