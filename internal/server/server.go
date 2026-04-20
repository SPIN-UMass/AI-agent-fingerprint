package server

import (
	"context"
	"crypto/tls"
	"fmt"
	"log"
	"net/http"

	"github.com/wi1dcard/fingerproxy/pkg/proxyserver"

	"agent-scraper/internal/content"
	"agent-scraper/internal/logging"
)

// Config holds server configuration.
type Config struct {
	HTTPSAddr  string
	HTTPAddr   string
	CertFile   string
	KeyFile    string
	LogDir     string
	ContentDir string
}

// Run starts the HTTPS fingerprinting server (and optional HTTP redirect).
func Run(ctx context.Context, cfg Config) error {
	logger, err := logging.NewLogger(cfg.LogDir)
	if err != nil {
		return fmt.Errorf("init logger: %w", err)
	}
	defer logger.Close()

	// ── Router ───────────────────────────────────────────────────────────────
	mux := http.NewServeMux()
	mux.Handle("/collect", CollectHandler(logger))
	mux.Handle("/", content.Handler(cfg.ContentDir))

	handler := LoggingHandler(mux, logger)

	// ── HTTPS SERVER ─────────────────────────────────────────────────────────
	tlsConf := &tls.Config{
		MinVersion: tls.VersionTLS12,
		NextProtos: []string{"h2", "http/1.1"},
	}

	httpsServer := &http.Server{
		Addr:      cfg.HTTPSAddr,
		Handler:   handler,
		TLSConfig: tlsConf,
	}

	// ── HTTP REDIRECT SERVER ────────────────────────────────────────────────
	if cfg.HTTPAddr != "" {
		go func() {
			log.Printf("HTTP redirect server listening on %s", cfg.HTTPAddr)
			err := http.ListenAndServe(cfg.HTTPAddr, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				target := "https://" + r.Host + r.URL.RequestURI()
				http.Redirect(w, r, target, http.StatusMovedPermanently)
			}))
			if err != nil {
				log.Printf("HTTP redirect error: %v", err)
			}
		}()
	}

	// ── GRACEFUL SHUTDOWN ───────────────────────────────────────────────────
	go func() {
		<-ctx.Done()
		log.Println("Shutting down servers...")
		httpsServer.Shutdown(context.Background())
	}()

	log.Printf("HTTPS server listening on %s", cfg.HTTPSAddr)

	// 🔥 THIS is the key fix:
	return httpsServer.ListenAndServeTLS(cfg.CertFile, cfg.KeyFile)
}

func runHTTPRedirect(listenAddr, httpsAddr string) {
	redirect := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		target := "https://" + r.Host + r.URL.RequestURI()
		http.Redirect(w, r, target, http.StatusMovedPermanently)
	})
	log.Printf("HTTP redirect server listening on %s", listenAddr)
	if err := http.ListenAndServe(listenAddr, redirect); err != nil {
		log.Printf("HTTP redirect server error: %v", err)
	}
}
