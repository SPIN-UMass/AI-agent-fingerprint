package server

import (
	"crypto/sha256"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/wi1dcard/fingerproxy/pkg/metadata"

	"agent-scraper/internal/capture"
	"agent-scraper/internal/logging"
)

// LoggingHandler wraps a content handler and logs fingerprint data for every request.
func LoggingHandler(content http.Handler, logger *logging.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received := time.Now()

		entry := &logging.RequestLog{
			ID:              uuid.NewString(),
			Timestamp:       received,
			RequestReceived: received,
			RemoteAddr:      r.RemoteAddr,
		}

		// Parse source IP/port
		if host, port, err := net.SplitHostPort(r.RemoteAddr); err == nil {
			entry.SourceIP = host
			entry.SourcePort = port
		}

		// TLS + HTTP/2 fingerprints from fingerproxy metadata
		if data, ok := metadata.FromContext(r.Context()); ok {
			entry.TLS = capture.ExtractTLS(data)
			entry.HTTP2 = capture.ExtractHTTP2(data)
		}

		// HTTP info
		entry.HTTP = buildHTTPInfo(r)

		// Serve the actual content
		content.ServeHTTP(w, r)

		// Timing
		entry.ResponseSent = time.Now()
		entry.ProcessingTimeMs = float64(entry.ResponseSent.Sub(received).Microseconds()) / 1000.0

		// Write log entry
		if err := logger.Write(entry); err != nil {
			log.Printf("error writing log: %v", err)
		}
	})
}

func buildHTTPInfo(r *http.Request) logging.HTTPInfo {
	info := logging.HTTPInfo{
		Method:   r.Method,
		Path:     r.URL.Path,
		Query:    r.URL.RawQuery,
		Protocol: r.Proto,
		Host:     r.Host,
		Headers:  make(map[string][]string),
	}

	// Ordered headers
	for _, key := range headerOrder(r.Header) {
		for _, val := range r.Header[key] {
			info.HeadersOrdered = append(info.HeadersOrdered, logging.HeaderPair{
				Name:  key,
				Value: val,
			})
		}
		info.Headers[key] = r.Header[key]
	}

	// Body hash
	if r.Body != nil {
		h := sha256.New()
		n, _ := io.Copy(h, r.Body)
		info.BodySize = n
		if n > 0 {
			info.BodySHA256 = fmt.Sprintf("%x", h.Sum(nil))
		}
	}

	// Extract common headers
	info.UserAgent = r.UserAgent()
	info.Accept = r.Header.Get("Accept")
	info.AcceptLang = r.Header.Get("Accept-Language")
	info.AcceptEnc = r.Header.Get("Accept-Encoding")
	info.SecChUA = r.Header.Get("Sec-Ch-Ua")
	info.SecChMobile = r.Header.Get("Sec-Ch-Ua-Mobile")
	info.SecChPlatf = r.Header.Get("Sec-Ch-Ua-Platform")
	info.SecFetchDest = r.Header.Get("Sec-Fetch-Dest")
	info.SecFetchMode = r.Header.Get("Sec-Fetch-Mode")
	info.SecFetchSite = r.Header.Get("Sec-Fetch-Site")
	info.SecFetchUser = r.Header.Get("Sec-Fetch-User")
	info.Cookie = r.Header.Get("Cookie")
	info.Referer = r.Header.Get("Referer")

	return info
}

// headerOrder returns header keys in iteration order.
// Go maps don't preserve insertion order, but for HTTP/2 requests
// fingerproxy's forked http2 package preserves order in the Header map.
func headerOrder(h http.Header) []string {
	seen := make(map[string]bool)
	var order []string
	for key := range h {
		if !seen[key] {
			seen[key] = true
			order = append(order, key)
		}
	}
	return order
}
