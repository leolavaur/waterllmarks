{
  inputs = { nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11"; };

  outputs = { nixpkgs, ... }:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems
        (system: f { pkgs = import nixpkgs { inherit system; }; });
    in {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShellNoCC {

          env = { VSCODE_PYTHON_PATH = ".venv/bin/python"; };

          packages = with pkgs; [
            uv

            texliveFull # for latex plots

            nixpkgs-fmt
            nixfmt
            nixd
          ];
        };
      });
    };

}
