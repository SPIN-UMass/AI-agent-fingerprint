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

	// TLS config
	cert, err := tls.LoadX509KeyPair(cfg.CertFile, cfg.KeyFile)
	if err != nil {
		return fmt.Errorf("load TLS cert: %w", err)
	}

	tlsConf := &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{"h2", "http/1.1"},
	}

	// ── Router ───────────────────────────────────────────────────────────────
	//
	// /collect  — receives sendBeacon payloads from logger.js (POST only).
	//             Written to interactions-YYYY-MM-DD.jsonl in LogDir.
	//             This endpoint is still wrapped by LoggingHandler so the
	//             HTTP/TLS fingerprint of the final unload request is also
	//             captured — useful for comparing against mid-session requests.
	//
	// /*        — serves static website-content files.
	//
	mux := http.NewServeMux()
	mux.Handle("/collect", CollectHandler(cfg.LogDir))
	mux.Handle("/", content.Handler(cfg.ContentDir))

	// Wrap the whole mux with fingerprint logging
	handler := LoggingHandler(mux, logger)

	// Start HTTP → HTTPS redirect server
	if cfg.HTTPAddr != "" {
		go runHTTPRedirect(cfg.HTTPAddr, cfg.HTTPSAddr)
	}

	// Start fingerproxy HTTPS server
	server := proxyserver.NewServer(ctx, handler, tlsConf)
	log.Printf("HTTPS server listening on %s", cfg.HTTPSAddr)
	return server.ListenAndServe(cfg.HTTPSAddr)
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