{ pkgs, ... }:

pkgs.mkShellNoCC {

  buildInputs = with pkgs; [
    (python312.withPackages (ps: [ ps.openai ]))
  ];
}