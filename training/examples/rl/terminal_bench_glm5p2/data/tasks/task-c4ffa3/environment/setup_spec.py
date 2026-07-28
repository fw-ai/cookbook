#!/usr/bin/env python3
"""Generate ESV Validation Script Framework (VSF) specification files.

Creates validation tree JSON files and rule script JSON files that define
the NIST ESV protocol payload validation logic.
"""
import json
import os


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


SPEC = "/app/spec"
RULES = f"{SPEC}/rules"
TREES = f"{SPEC}/trees"

print("Generating ESV validation spec files...")

# ============================================================
# Common Rules
# ============================================================

write_json(f"{RULES}/CommonRules/notNull.json", {
    "vsfScript": [{
        "lineType": "Rule",
        "ruleType": "External",
        "parameters": {"ruleText": "currentProperty != null"}
    }]
})

write_json(f"{RULES}/CommonRules/notNullAndNotWhitespace.json", {
    "vsfScript": [{
        "lineType": "Rule",
        "ruleType": "External",
        "parameters": {
            "ruleText": "! string.IsNullOrWhiteSpace(currentProperty)"
        }
    }]
})

write_json(f"{RULES}/CommonRules/listMinCount.json", {
    "vsfScript": [
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNull.json"}
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "description": "At least 1 list item is required",
            "parameters": {"ruleText": "currentProperty.Count() > 0"}
        }
    ]
})

write_json(f"{RULES}/CommonRules/listIsUnique.json", {
    "vsfScript": [{
        "lineType": "Rule",
        "ruleType": "External",
        "parameters": {
            "ruleText":
                "currentProperty.Distinct().Count() == currentProperty.Count()"
        }
    }]
})

write_json(f"{RULES}/CommonRules/validObjectId.json", {
    "vsfScript": [{
        "lineType": "Rule",
        "ruleType": "External",
        "description": "Object IDs are greater than 0",
        "parameters": {"ruleText": "currentProperty > 0"}
    }]
})

# ============================================================
# Register Request — top-level field rules
# ============================================================

write_json(f"{RULES}/Rules/RegisterRequest/primaryNoiseSource.json", {
    "vsfScript": [
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNull.json"}
        },
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNullAndNotWhitespace.json"}
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "parameters": {"ruleText": "currentProperty.Length <= 64"}
        }
    ]
})

write_json(f"{RULES}/Rules/RegisterRequest/bitsPerSample.json", {
    "vsfScript": [
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNull.json"}
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "parameters": {
                "ruleText": "currentProperty >= 1 && currentProperty <= 256"
            }
        }
    ]
})

write_json(f"{RULES}/Rules/RegisterRequest/hMinEstimate.json", {
    "vsfScript": [
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNull.json"}
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "parameters": {
                "ruleText":
                    "currentProperty >= 0.0 && currentProperty <= "
                    "parentProperty.bitsPerSample"
            }
        }
    ]
})

write_json(f"{RULES}/Rules/RegisterRequest/Shared/minRestart.json", {
    "vsfScript": [
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNull.json"}
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "parameters": {"ruleText": "currentProperty >= 1000"}
        }
    ]
})

# ============================================================
# Register Request — Conditioning Component rules
# ============================================================

write_json(
    f"{RULES}/Rules/RegisterRequest/ConditioningComponent/minHin.json", {
        "vsfScript": [{
            "lineType": "Rule",
            "ruleType": "External",
            "parameters": {"ruleText": "currentProperty >= 0"}
        }]
    })

# -- Vetted conditioning component rules --

write_json(
    f"{RULES}/Rules/RegisterRequest/ConditioningComponent"
    "/Vetted/bijectiveClaimIsNotApplicable.json",
    {
        "vsfScript": [{
            "lineType": "Branch",
            "conditions": [{
                "if": {
                    "scriptLines": [{
                        "lineType": "Rule",
                        "ruleType": "External",
                        "parameters": {
                            "ruleText": "parentProperty.vetted == true"
                        }
                    }]
                },
                "then": {
                    "scriptLines": [{
                        "lineType": "Rule",
                        "ruleType": "External",
                        "parameters": {
                            "ruleText": "currentProperty == null"
                        }
                    }]
                }
            }]
        }]
    })

write_json(
    f"{RULES}/Rules/RegisterRequest/ConditioningComponent"
    "/Vetted/validationNumber.json",
    {
        "vsfScript": [{
            "lineType": "Branch",
            "conditions": [{
                "if": {
                    "scriptLines": [{
                        "lineType": "Rule",
                        "ruleType": "External",
                        "parameters": {
                            "ruleText": "parentProperty.vetted == true"
                        }
                    }]
                },
                "then": {
                    "scriptLines": [
                        {
                            "lineType": "ImportScript",
                            "parameters": {
                                "scriptFile": "CommonRules/notNull.json"
                            }
                        },
                        {
                            "lineType": "ImportScript",
                            "parameters": {
                                "scriptFile":
                                    "CommonRules/notNullAndNotWhitespace.json"
                            }
                        }
                    ]
                }
            }]
        }]
    })

# -- Non-vetted conditioning component rules --

write_json(
    f"{RULES}/Rules/RegisterRequest/ConditioningComponent"
    "/NonVetted/isBijectiveClaim.json",
    {
        "vsfScript": [{
            "lineType": "Branch",
            "conditions": [{
                "if": {
                    "scriptLines": [{
                        "lineType": "Rule",
                        "ruleType": "External",
                        "parameters": {
                            "ruleText": "parentProperty.vetted == false"
                        }
                    }]
                },
                "then": {
                    "scriptLines": [{
                        "lineType": "ImportScript",
                        "parameters": {
                            "scriptFile": "CommonRules/notNull.json"
                        }
                    }]
                }
            }]
        }]
    })

write_json(
    f"{RULES}/Rules/RegisterRequest/ConditioningComponent"
    "/NonVetted/validationNumberIsNotApplicable.json",
    {
        "vsfScript": [{
            "lineType": "Branch",
            "conditions": [{
                "if": {
                    "scriptLines": [{
                        "lineType": "Rule",
                        "ruleType": "External",
                        "parameters": {
                            "ruleText": "parentProperty.vetted == false"
                        }
                    }]
                },
                "then": {
                    "scriptLines": [{
                        "lineType": "Rule",
                        "ruleType": "External",
                        "parameters": {
                            "ruleText": "currentProperty == null"
                        }
                    }]
                }
            }]
        }]
    })

# ============================================================
# Certify Request rules
# ============================================================

write_json(f"{RULES}/Rules/CertifyRequest/entropyId.json", {
    "vsfScript": [
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNull.json"}
        },
        {
            "lineType": "ImportScript",
            "parameters": {
                "scriptFile": "CommonRules/notNullAndNotWhitespace.json"
            }
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "parameters": {"ruleText": "currentProperty.Length == 4"}
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "description":
                "EntropyId should only contain alphanumeric characters.",
            "parameters": {
                "ruleText":
                    "Regex.IsMatch(currentProperty, \"^[a-zA-Z0-9]+$\")"
            }
        },
        {
            "lineType": "State",
            "parameters": {
                "key": "upperId",
                "value": "currentProperty.ToUpper()"
            }
        },
        {
            "lineType": "Information",
            "description": "EID - Entropy ID used to identify the review",
            "parameters": {
                "ruleText":
                    "String.Format(\"Entropy ID: {0}\", upperId)"
            }
        }
    ]
})

write_json(f"{RULES}/Rules/CertifyRequest/moduleId.json", {
    "vsfScript": [
        {
            "lineType": "ImportScript",
            "parameters": {"scriptFile": "CommonRules/notNull.json"}
        },
        {
            "lineType": "Rule",
            "ruleType": "External",
            "parameters": {"ruleText": "currentProperty > 0"}
        }
    ]
})

write_json(
    f"{RULES}/Rules/CertifyRequest"
    "/EntropyAssessmentReference/eaIdIsDistinct.json",
    {
        "vsfScript": [
            {
                "lineType": "State",
                "parameters": {
                    "key": "eaIds",
                    "value":
                        "ExtractField(currentProperty, \"eaId\")"
                }
            },
            {
                "lineType": "Rule",
                "ruleType": "External",
                "parameters": {
                    "ruleText":
                        "eaIds.Distinct().Count() == eaIds.Count()"
                }
            }
        ]
    })

# ============================================================
# Validation Trees
# ============================================================

register_tree = [{
    "modelName": "EntropyAssessmentRegisterPayload",
    "rootNode": {
        "nodeData": {
            "vsfScriptFiles": [
                {"scriptFile": "CommonRules/notNull.json"}
            ]
        },
        "nodes": [
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "primaryNoiseSource"},
                "nodeData": {
                    "vsfScriptFiles": [{
                        "scriptFile":
                            "Rules/RegisterRequest/primaryNoiseSource.json"
                    }]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "iidClaim"},
                "nodeData": {
                    "vsfScriptFiles": [
                        {"scriptFile": "CommonRules/notNull.json"}
                    ]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "bitsPerSample"},
                "nodeData": {
                    "vsfScriptFiles": [{
                        "scriptFile":
                            "Rules/RegisterRequest/bitsPerSample.json"
                    }]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "hminEstimate"},
                "nodeData": {
                    "vsfScriptFiles": [{
                        "scriptFile":
                            "Rules/RegisterRequest/hMinEstimate.json"
                    }]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "physical"},
                "nodeData": {
                    "vsfScriptFiles": [
                        {"scriptFile": "CommonRules/notNull.json"}
                    ]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "additionalNoiseSources"},
                "nodeData": {
                    "vsfScriptFiles": [
                        {"scriptFile": "CommonRules/notNull.json"}
                    ]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "numberOfRestarts"},
                "nodeData": {
                    "vsfScriptFiles": [{
                        "scriptFile":
                            "Rules/RegisterRequest/Shared/minRestart.json"
                    }]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "samplesPerRestart"},
                "nodeData": {
                    "vsfScriptFiles": [{
                        "scriptFile":
                            "Rules/RegisterRequest/Shared/minRestart.json"
                    }]
                }
            },
            {
                "nodeType": "list",
                "property": {
                    "internalIdentifier": "conditioningComponent"
                },
                "nodeData": {
                    "vsfScriptFiles": [
                        {"scriptFile": "CommonRules/listMinCount.json"}
                    ]
                },
                "listItem": {
                    "branchNodeData": {
                        "runBeforeListItem": {
                            "vsfScriptFiles": [
                                {"scriptFile": "CommonRules/notNull.json"}
                            ]
                        }
                    },
                    "nodes": [
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "vetted"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"}
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "bijectiveClaim"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {
                                        "scriptFile":
                                            "Rules/RegisterRequest/"
                                            "ConditioningComponent/Vetted/"
                                            "bijectiveClaimIsNotApplicable"
                                            ".json"
                                    },
                                    {
                                        "scriptFile":
                                            "Rules/RegisterRequest/"
                                            "ConditioningComponent/NonVetted"
                                            "/isBijectiveClaim.json"
                                    }
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "validationNumber"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {
                                        "scriptFile":
                                            "Rules/RegisterRequest/"
                                            "ConditioningComponent/Vetted/"
                                            "validationNumber.json"
                                    },
                                    {
                                        "scriptFile":
                                            "Rules/RegisterRequest/"
                                            "ConditioningComponent/NonVetted"
                                            "/validationNumberIsNot"
                                            "Applicable.json"
                                    }
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "minNin"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {"scriptFile":
                                        "CommonRules/validObjectId.json"}
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "minHin"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {
                                        "scriptFile":
                                            "Rules/RegisterRequest/"
                                            "ConditioningComponent/"
                                            "minHin.json"
                                    }
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "nw"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {"scriptFile":
                                        "CommonRules/validObjectId.json"}
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "nOut"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {"scriptFile":
                                        "CommonRules/validObjectId.json"}
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "hOut"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"}
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
}]

certify_tree = [{
    "modelName": "CertifyRequestPayloadFullSubmission",
    "rootNode": {
        "nodeData": {
            "vsfScriptFiles": [
                {"scriptFile": "CommonRules/notNull.json"}
            ]
        },
        "nodes": [
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "entropyId"},
                "nodeData": {
                    "vsfScriptFiles": [{
                        "scriptFile":
                            "Rules/CertifyRequest/entropyId.json"
                    }]
                }
            },
            {
                "nodeType": "leaf",
                "property": {
                    "internalIdentifier":
                        "limitEntropyAssessmentToSingleModule"
                },
                "nodeData": {
                    "vsfScriptFiles": [
                        {"scriptFile": "CommonRules/notNull.json"}
                    ]
                }
            },
            {
                "nodeType": "leaf",
                "property": {"internalIdentifier": "moduleId"},
                "nodeData": {
                    "vsfScriptFiles": [{
                        "scriptFile":
                            "Rules/CertifyRequest/moduleId.json"
                    }]
                }
            },
            {
                "nodeType": "list",
                "property": {
                    "internalIdentifier": "supportingDocumentation"
                },
                "nodeData": {
                    "vsfScriptFiles": [
                        {"scriptFile": "CommonRules/listMinCount.json"}
                    ]
                },
                "listItem": {
                    "branchNodeData": {
                        "runBeforeListItem": {
                            "vsfScriptFiles": [
                                {"scriptFile": "CommonRules/notNull.json"}
                            ]
                        }
                    },
                    "nodes": [
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "sdId"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {"scriptFile":
                                        "CommonRules/validObjectId.json"}
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "accessToken"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {
                                        "scriptFile":
                                            "CommonRules/"
                                            "notNullAndNotWhitespace.json"
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "nodeType": "list",
                "property": {
                    "internalIdentifier": "entropyAssessments"
                },
                "nodeData": {
                    "vsfScriptFiles": [
                        {"scriptFile": "CommonRules/listMinCount.json"},
                        {
                            "scriptFile":
                                "Rules/CertifyRequest/"
                                "EntropyAssessmentReference/"
                                "eaIdIsDistinct.json"
                        }
                    ]
                },
                "listItem": {
                    "branchNodeData": {
                        "runBeforeListItem": {
                            "vsfScriptFiles": [
                                {"scriptFile": "CommonRules/notNull.json"}
                            ]
                        }
                    },
                    "nodes": [
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "eaId"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {"scriptFile":
                                        "CommonRules/validObjectId.json"}
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "oeId"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {"scriptFile":
                                        "CommonRules/validObjectId.json"}
                                ]
                            }
                        },
                        {
                            "nodeType": "leaf",
                            "property": {
                                "internalIdentifier": "accessToken"
                            },
                            "nodeData": {
                                "vsfScriptFiles": [
                                    {"scriptFile":
                                        "CommonRules/notNull.json"},
                                    {
                                        "scriptFile":
                                            "CommonRules/"
                                            "notNullAndNotWhitespace.json"
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
}]

write_json(f"{TREES}/registerEntropySource.json", register_tree)
write_json(f"{TREES}/certifyFull.json", certify_tree)

print("Done. Spec files at", SPEC)
