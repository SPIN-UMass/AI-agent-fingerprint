package logging

import "time"

// RequestLog is the JSONL schema for each captured request.
type RequestLog struct {
	// Meta
	ID        string    `json:"id"`
	Timestamp time.Time `json:"timestamp"`

	// Network
	RemoteAddr string `json:"remote_addr"`
	SourceIP   string `json:"source_ip"`
	SourcePort string `json:"source_port"`

	// TLS
	TLS *TLSInfo `json:"tls,omitempty"`

	// HTTP/2
	HTTP2 *HTTP2Info `json:"http2,omitempty"`

	// HTTP
	HTTP HTTPInfo `json:"http"`

	// Timing
	RequestReceived  time.Time `json:"request_received"`
	ResponseSent     time.Time `json:"response_sent"`
	ProcessingTimeMs float64   `json:"processing_time_ms"`
}

type TLSInfo struct {
	Version            string `json:"version"`
	CipherSuite        string `json:"cipher_suite"`
	NegotiatedProtocol string `json:"negotiated_protocol"`
	ServerName         string `json:"server_name"`

	// JA3
	JA3Hash string `json:"ja3_hash"`
	JA3Raw  string `json:"ja3_raw,omitempty"`

	// JA4
	JA4Hash string `json:"ja4_hash"`
	JA4Raw  string `json:"ja4_raw,omitempty"`

	// ClientHello details
	CipherSuitesOffered []uint16 `json:"cipher_suites_offered,omitempty"`
	ExtensionsOffered   []uint16 `json:"extensions_offered,omitempty"`
	EllipticCurves      []uint16 `json:"elliptic_curves,omitempty"`
	PointFormats        []uint8  `json:"point_formats,omitempty"`
	SignatureSchemes     []uint16 `json:"signature_schemes,omitempty"`
	SupportedVersions   []uint16 `json:"supported_versions,omitempty"`
	ALPNProtocols       []string `json:"alpn_protocols,omitempty"`
}

type HTTP2Info struct {
	AkamaiFingerprint string            `json:"akamai_fingerprint"`
	Settings          map[string]uint32 `json:"settings,omitempty"`
	WindowUpdateSize  uint32            `json:"window_update_size,omitempty"`
	PseudoHeaderOrder []string          `json:"pseudo_header_order,omitempty"`
}

type HTTPInfo struct {
	Method   string `json:"method"`
	Path     string `json:"path"`
	Query    string `json:"query,omitempty"`
	Protocol string `json:"protocol"`
	Host     string `json:"host"`

	// Headers as ordered list (preserves order for HTTP/2)
	HeadersOrdered []HeaderPair `json:"headers_ordered"`
	// Headers as map (convenient for analysis)
	Headers map[string][]string `json:"headers"`

	// Body info
	BodySHA256 string `json:"body_sha256,omitempty"`
	BodySize   int64  `json:"body_size"`

	// Common extracted headers
	UserAgent    string   `json:"user_agent,omitempty"`
	Accept       string   `json:"accept,omitempty"`
	AcceptLang   string   `json:"accept_language,omitempty"`
	AcceptEnc    string   `json:"accept_encoding,omitempty"`
	SecChUA      string   `json:"sec_ch_ua,omitempty"`
	SecChMobile  string   `json:"sec_ch_ua_mobile,omitempty"`
	SecChPlatf   string   `json:"sec_ch_ua_platform,omitempty"`
	SecFetchDest string   `json:"sec_fetch_dest,omitempty"`
	SecFetchMode string   `json:"sec_fetch_mode,omitempty"`
	SecFetchSite string   `json:"sec_fetch_site,omitempty"`
	SecFetchUser string   `json:"sec_fetch_user,omitempty"`
	Cookie       string   `json:"cookie,omitempty"`
	Referer      string   `json:"referer,omitempty"`
}

type HeaderPair struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}
