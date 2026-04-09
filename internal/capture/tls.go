package capture

import (
	"crypto/tls"
	"fmt"

	utls "github.com/refraction-networking/utls"
	"github.com/wi1dcard/fingerproxy/pkg/fingerprint"
	"github.com/wi1dcard/fingerproxy/pkg/metadata"

	"agent-scraper/internal/logging"
)

func tlsVersionName(v uint16) string {
	switch v {
	case tls.VersionTLS10:
		return "TLS 1.0"
	case tls.VersionTLS11:
		return "TLS 1.1"
	case tls.VersionTLS12:
		return "TLS 1.2"
	case tls.VersionTLS13:
		return "TLS 1.3"
	default:
		return fmt.Sprintf("0x%04x", v)
	}
}

func cipherSuiteName(id uint16) string {
	if name := tls.CipherSuiteName(id); name != "" {
		return name
	}
	return fmt.Sprintf("0x%04x", id)
}

// ExtractTLS extracts TLS fingerprint info from fingerproxy metadata.
func ExtractTLS(data *metadata.Metadata) *logging.TLSInfo {
	cs := data.ConnectionState

	info := &logging.TLSInfo{
		Version:            tlsVersionName(cs.Version),
		CipherSuite:        cipherSuiteName(cs.CipherSuite),
		NegotiatedProtocol: cs.NegotiatedProtocol,
		ServerName:         cs.ServerName,
	}

	// JA3
	if ja3, err := fingerprint.JA3Fingerprint(data); err == nil {
		info.JA3Hash = ja3
	}

	// JA4
	if ja4, err := fingerprint.JA4Fingerprint(data); err == nil {
		info.JA4Hash = ja4
	}

	// Parse raw ClientHello for detailed fields
	parseClientHello(data.ClientHelloRecord, info)

	return info
}

func parseClientHello(record []byte, info *logging.TLSInfo) {
	if len(record) == 0 {
		return
	}

	chs := &utls.ClientHelloSpec{}
	err := chs.FromRaw(record, true)
	if err != nil {
		return
	}

	// Cipher suites
	for _, suite := range chs.CipherSuites {
		info.CipherSuitesOffered = append(info.CipherSuitesOffered, suite)
	}

	// Walk extensions for details
	for _, ext := range chs.Extensions {
		switch e := ext.(type) {
		case *utls.SupportedCurvesExtension:
			for _, c := range e.Curves {
				info.EllipticCurves = append(info.EllipticCurves, uint16(c))
			}
		case *utls.SupportedPointsExtension:
			info.PointFormats = append(info.PointFormats, e.SupportedPoints...)
		case *utls.SignatureAlgorithmsExtension:
			for _, s := range e.SupportedSignatureAlgorithms {
				info.SignatureSchemes = append(info.SignatureSchemes, uint16(s))
			}
		case *utls.SupportedVersionsExtension:
			for _, v := range e.Versions {
				info.SupportedVersions = append(info.SupportedVersions, v)
			}
		case *utls.ALPNExtension:
			info.ALPNProtocols = append(info.ALPNProtocols, e.AlpnProtocols...)
		}
	}
}
