# Stage 1: Build
FROM golang:1.25-alpine AS builder

WORKDIR /app

# Download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN go build -o agent-scraper ./cmd/agent-scraper

# Stage 2: Run
FROM alpine:3.20

# Install certificates for HTTPS
RUN apk add --no-cache ca-certificates

WORKDIR /app

# Copy built binary and other files
COPY --from=builder /app/agent-scraper .
COPY configs ./configs
COPY keys ./keys
COPY tls.crt .
COPY tls.key .
COPY website-content ./website-content

# Ensure logs directory exists
RUN mkdir -p /app/logs && chmod 755 /app/logs

# Make the binary executable
RUN chmod +x agent-scraper

# Expose /app/logs as a volume for persistence
VOLUME ["/app/logs"]

# Default entrypoint with explicit log-dir
ENTRYPOINT ["./agent-scraper", "--content-dir", "./website-content", "--log-dir", "/app/logs"]