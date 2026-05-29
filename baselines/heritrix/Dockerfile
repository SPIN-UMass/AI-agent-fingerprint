# Heritrix 3 baseline crawler image for the SPIN AI-agent-fingerprint study.
#
# There is NO official Heritrix Docker image (Heritrix is an Internet Archive
# project, not Apache). We build a small pinned image: Temurin JDK 17 (Heritrix
# >= 3.15.0 requires Java 17 because Spring Framework 6.1 does) plus the official
# Heritrix distribution zip pulled from Maven Central at build time and unzipped
# INSIDE the image (never vendored into the repo).
#
# Pinned version: 3.15.0  (latest release as of 2026-05-26; org.archive.heritrix:heritrix)
#   https://repo1.maven.org/maven2/org/archive/heritrix/heritrix/3.15.0/heritrix-3.15.0-dist.zip
#   sha1: bc25b1ec3fe030ae74d42909b9ea599d08204564
FROM eclipse-temurin:17-jdk

ARG HERITRIX_VERSION=3.15.0
ARG HERITRIX_SHA1=bc25b1ec3fe030ae74d42909b9ea599d08204564
ENV HERITRIX_HOME=/opt/heritrix
ENV FOREGROUND=true

# unzip to extract the dist; curl to download it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Download + verify + unzip the official Heritrix dist INSIDE the image build.
RUN set -eux; \
    url="https://repo1.maven.org/maven2/org/archive/heritrix/heritrix/${HERITRIX_VERSION}/heritrix-${HERITRIX_VERSION}-dist.zip"; \
    curl -fSL "$url" -o /tmp/heritrix.zip; \
    echo "${HERITRIX_SHA1}  /tmp/heritrix.zip" | sha1sum -c -; \
    unzip -q /tmp/heritrix.zip -d /opt; \
    mv "/opt/heritrix-${HERITRIX_VERSION}" "${HERITRIX_HOME}"; \
    rm -f /tmp/heritrix.zip; \
    chmod +x "${HERITRIX_HOME}/bin/heritrix"

WORKDIR ${HERITRIX_HOME}

# Web UI / REST API (HTTPS, self-signed, DIGEST auth).
EXPOSE 8443

# -a admin:admin  : Web UI / REST credentials.
# -b /            : bind the admin server to ALL interfaces (default is localhost
#                   only, which would be unreachable through the docker port map).
# Credentials are passed at runtime by run.sh so they are not baked into the image.
ENTRYPOINT ["bin/heritrix"]
CMD ["-a", "admin:admin", "-b", "/"]
