package logging

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Logger writes RequestLog entries as append-only JSONL.
type Logger struct {
	mu   sync.Mutex
	file *os.File
	enc  *json.Encoder
	dir  string
}

// WriteInteraction logs a single interaction batch from logger.js. Thread-safe.
func (l *Logger) WriteInteraction(entry *InteractionLog) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Use a separate interactions file alongside the requests file
	path := filepath.Join(l.dir, fmt.Sprintf("interactions-%s.jsonl", time.Now().Format("2006-01-02")))
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open interactions log: %w", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	return enc.Encode(entry)
}

// NewLogger creates a JSONL logger writing to the given directory.
// Log files are named by date: requests-YYYY-MM-DD.jsonl
func NewLogger(dir string) (*Logger, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create log dir: %w", err)
	}
	l := &Logger{dir: dir}
	if err := l.openFile(); err != nil {
		return nil, err
	}
	return l, nil
}

func (l *Logger) logFileName() string {
	return filepath.Join(l.dir, fmt.Sprintf("requests-%s.jsonl", time.Now().Format("2006-01-02")))
}

func (l *Logger) openFile() error {
	f, err := os.OpenFile(l.logFileName(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open log file: %w", err)
	}
	l.file = f
	l.enc = json.NewEncoder(f)
	l.enc.SetEscapeHTML(false)
	return nil
}

// Write logs a single request entry. Thread-safe.
func (l *Logger) Write(entry *RequestLog) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Rotate to new file if date changed
	expected := l.logFileName()
	if l.file.Name() != expected {
		l.file.Close()
		if err := l.openFile(); err != nil {
			return err
		}
	}

	return l.enc.Encode(entry)
}

// Close closes the underlying file.
func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}
