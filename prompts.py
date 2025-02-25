
CAPTION_GEN_PROMPT = '''
You are a creative assistant who strictly following the rules, tasked with designing realistic and practical novel concepts that satisfy given conditions and generating concise descriptions of these concepts.

Generate ten distinct descriptions of novel concepts, each featuring visible, tangible characteristics suitable for image captions. The descriptions should focus on visually representable details and avoid to include non-visual characteristics such as sound, smell, and taste. Each description must consist of at most three sentences that are simple and concise and should integrate multiple functions into a cohesive and unique design without simply combining existing objects. Each concept must incorporate all the given positive constraints and avoid similarities with the given negative constraints. The ten descriptions must be clearly different from one another, avoiding redundancy and unnecessary explanations. You must avoid to use ambiguous and abstract words such as "futuristic", "compact", "innovative", "modern", "dynamic", "automatic" or "versatile,".

Please separate each description with '\n\n'. Simply follow the format given in the example below.
Positive Constraints: [sit, store]
Negative Constraints: [chair, car, sofa, bench, shelve, drawer]
Image Captions: ["A spherical pod seat with a cushioned top, rotating metallic shell, integrated storage slots, and soft LED lighting along the edges.\n\nAn oval-shaped seat with a central storage that unfolds like petals, featuring a cushioned matte finish.\n\nA tall spiral tower seat with a cushioned top, metallic surface, and hidden storage spaces revealed by twisting the structure.\n\nA spherical acrylic seat with a cushioned interior, a pull-out drawer at the base, and a clamshell top that rotates open. Grooves on the surface reflect light for a sleek appearance.\n\nA crescent-shaped seat with a fabric backrest, shelving underneath, and storage pockets in the curved arms. The base is polished wood.\n\nA hexagonal pod seat with sliding panels at the base for storage, a honeycomb-patterned cushion, and a metallic frame with rubberized corners.\n\nA cylindrical seat with a cushioned top, spiral-textured sides, and sliding side panels for storage. The base is slightly raised by a ring frame.\n\nA trapezoidal seat with a slanted backrest, a soft cushion, and a visible side compartment for storage. Metallic trim edges double as handles, and the base is perforated.\n\nA cube seat with a removable cushioned top and recessed side grips for storage access. The exterior is matte with polished metallic corners.\n\nA pill-shaped seat with velvet-like cushions and sliding side drawers for storage. The base has silicone pads for stability."]
'''

CAPTION_TEST_GEN_PROMPT = '''
You are a creative assistant who strictly following the rules, tasked with designing realistic and practical novel concepts that satisfy given conditions and generating concise descriptions of these concepts.

Generate a single description of novel concept, featuring visible, tangible characteristics suitable for image captions. The description should focus on visually representable details and avoid to include non-visual characteristics such as sound, smell, and taste. Each description must consist of at most three sentences that are simple and concise and should integrate multiple functions into a cohesive and unique design without simply combining existing objects. Each concept must incorporate all the given positive constraints and avoid similarities with the given negative constraints. The description must avoid redundancy and unnecessary explanations. You must avoid to use ambiguous and abstract words such as "futuristic", "compact", "innovative", "modern", "dynamic", "automatic" or "versatile,".

Simply follow the format given in the example below.
Positive Constraints: [sit, store]
Negative Constraints: [chair, car, sofa, bench, shelve, drawer]
Image Caption: "A spherical pod seat with a cushioned top, rotating metallic shell, integrated storage slots, and soft LED lighting along the edges."
'''

