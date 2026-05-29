# Scrapy baseline crawler environment.
#
# Usage:
#   cd baselines/scrapy
#   nix-shell --run "scrapy version"
#   nix-shell --run "./run.sh"
#
# Pins to the channel's python3Packages.scrapy (currently Scrapy 2.14.1).
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3Packages.scrapy
  ];
}
