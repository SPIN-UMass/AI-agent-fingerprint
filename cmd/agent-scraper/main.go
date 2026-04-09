package main

import (
	"context"
	"flag"
	"log"
	"os/signal"
	"syscall"

	"agent-scraper/internal/server"
)

func main() {
	httpsAddr := flag.String("https", ":443", "HTTPS listen address")
	httpAddr := flag.String("http", ":80", "HTTP redirect listen address (empty to disable)")
	certFile := flag.String("cert", "tls.crt", "TLS certificate file")
	keyFile := flag.String("key", "tls.key", "TLS private key file")
	logDir := flag.String("log-dir", "logs", "Directory for JSONL log files")
	contentDir := flag.String("content-dir", "website-content", "Directory to serve static files from (empty for default page)")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	cfg := server.Config{
		HTTPSAddr:  *httpsAddr,
		HTTPAddr:   *httpAddr,
		CertFile:   *certFile,
		KeyFile:    *keyFile,
		LogDir:     *logDir,
		ContentDir: *contentDir,
	}

	if err := server.Run(ctx, cfg); err != nil {
		log.Fatal(err)
	}
}
