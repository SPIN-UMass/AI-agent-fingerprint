package capture

import (
	"fmt"

	"github.com/wi1dcard/fingerproxy/pkg/metadata"

	"agent-scraper/internal/logging"
)

// HTTP/2 SETTINGS IDs → human-readable names
var settingNames = map[uint16]string{
	1: "HEADER_TABLE_SIZE",
	2: "ENABLE_PUSH",
	3: "MAX_CONCURRENT_STREAMS",
	4: "INITIAL_WINDOW_SIZE",
	5: "MAX_FRAME_SIZE",
	6: "MAX_HEADER_LIST_SIZE",
}

func settingName(id uint16) string {
	if name, ok := settingNames[id]; ok {
		return name
	}
	return fmt.Sprintf("UNKNOWN_%d", id)
}

// ExtractHTTP2 extracts HTTP/2 fingerprint info from fingerproxy metadata.
// Returns nil if the connection is not HTTP/2.
func ExtractHTTP2(data *metadata.Metadata) *logging.HTTP2Info {
	if data.ConnectionState.NegotiatedProtocol != "h2" {
		return nil
	}

	frames := &data.HTTP2Frames

	info := &logging.HTTP2Info{
		AkamaiFingerprint: frames.String(),
		Settings:          make(map[string]uint32),
		WindowUpdateSize:  frames.WindowUpdateIncrement,
	}

	for _, s := range frames.Settings {
		info.Settings[settingName(s.Id)] = s.Val
	}

	// Extract pseudo-header order from HEADERS frame
	for _, h := range frames.Headers {
		if len(h.Name) >= 2 && h.Name[0] == ':' {
			info.PseudoHeaderOrder = append(info.PseudoHeaderOrder, h.Name)
		}
	}

	return info
}
