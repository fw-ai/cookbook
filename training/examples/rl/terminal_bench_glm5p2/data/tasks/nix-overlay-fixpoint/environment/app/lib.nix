# Configuration library functions
# Implements functional composition patterns for config management
rec {
  # Fixed-point combinator
  fix = f: let x = f x; in x;

  # Compose two overlays (extensions)
  # An overlay has the signature: self: super: { ... }
  # where self is the final result (fixpoint) and super is the previous layer
  composeExtensions = f: g: self: super:
    let
      fApplied = f self super;
      gApplied = g (super // fApplied) self;
    in
    fApplied // gApplied;

  # Apply a list of overlays to a base attribute set
  applyOverlays = base: overlays:
    let
      composed = builtins.foldl' composeExtensions (_: _: {}) overlays;
    in
    fix (self: base // composed self base);

  # Recursively merge two attribute sets
  # For nested attrsets: merge recursively
  # For lists: concatenate
  # For other values: b takes precedence
  recursiveUpdate = a: b:
    a // builtins.mapAttrs (name: bVal:
      if builtins.hasAttr name a && builtins.isAttrs a.${name} && builtins.isAttrs bVal
      then recursiveUpdate a.${name} bVal
      else bVal
    ) b;

  # Merge a list of attribute sets using recursiveUpdate
  foldAttrs = list:
    builtins.foldl' recursiveUpdate {} list;

  # Concatenate nested lists
  concatLists = builtins.concatLists;

  # Remove duplicates from a list, preserving first occurrence order
  unique = list:
    builtins.foldl' (acc: x: if builtins.elem x acc then acc else acc ++ [x]) [] list;

  # Sort a list of comparable values
  sort = builtins.sort builtins.lessThan;

  # Filter attributes by predicate
  filterAttrs = pred: attrs:
    builtins.listToAttrs (
      builtins.filter (x: pred x.name x.value)
        (map (name: { inherit name; value = attrs.${name}; })
          (builtins.attrNames attrs))
    );

  # Conditionally include attributes
  optionalAttrs = cond: attrs: if cond then attrs else {};
}
