package content

import (
	"net/http"
)

// Handler returns an HTTP handler that serves static files from dir.
// If dir is empty, serves a minimal default page.
func Handler(dir string) http.Handler {
	if dir != "" {
		return http.FileServer(http.Dir(dir))
	}

	// Fallback: minimal default page
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(`<!DOCTYPE html><html><head><title>Welcome</title></head><body><h1>Welcome</h1><p>Under construction.</p></body></html>`))
	})
}
