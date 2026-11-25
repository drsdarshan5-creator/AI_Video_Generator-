# Anant and the Dawn Dial - Production Shot List

## Plan summary

- **Story basis:** A continuity-focused development of the existing CineAI
  cosmic-odyssey POC.
- **Story:** Anant finds a sapphire prism in a ruined desert observatory,
  follows its light to an obsidian chamber, aligns it with an ancient dial,
  and returns to the salt basin at dawn.
- **Runtime:** 30 shots x 2.0 seconds = 60.0 seconds raw. Two 0.5-second
  crossfades subtract 1.0 second, for an estimated **59.0-second assembled
  video**. Other transitions are cuts with no overlap.
- **Generation fit:** Each request is 2.0 seconds. At the verified 8 FPS Fast
  Draft rate, this maps to 16 frames per shot.
- **Character lock:** Anant is the only character: fictional South Asian man in
  his early 30s, medium-brown skin, short black hair, neat beard, weathered
  indigo hooded coat, charcoal scarf, dark trousers, brown leather gloves, and
  a brass satchel. The sapphire prism remains with him after shot 8.
- **Look and environment:** Same reflective salt basin and sandstone
  observatory, red ridge, obsidian arch, and underground dial chamber; indigo
  twilight moves gradually to gold dawn. Dry salt wind, ridge dust, and still
  underground air; no abrupt weather changes. Grounded photorealistic fantasy,
  restrained motion, no text or logos.
- **Reference limitation:** The JSON's `required_previous_shot_reference`
  values are editorial shot IDs for continuity. They are not generated-media
  paths. The current CogVideoX text-to-video provider does not ingest previous
  video/image references, so each prompt repeats the character and prop
  anchors.
- The complete generation prompt and all 14 requested shot fields are recorded
  per shot in [anant_1minute_shots.json](./anant_1minute_shots.json), nested in
  `request.metadata` so the file is accepted by the existing `VideoBatchRunner`.

## Shot-by-shot review

| # | Duration | Scene and single action | Characters | Location / time / environment | Camera | Transition to next | Continuity / previous reference |
|---:|---:|---|---|---|---|---|---|
| 1 | 2.0 s | Anant takes one step toward the observatory. | Anant | Reflective salt basin; blue twilight, twin moons, dry salt wind. | Wide establishing; slow push-in. | Cut, 0.0 s. | Establish wardrobe, basin, moons, and direction. No previous reference. |
| 2 | 2.0 s | One boot presses a footprint into the salt crust. | Anant, lower body | Same basin; blue twilight, pale salt dust. | Low tracking insert; left-to-right. | Cut, 0.0 s. | Match shot 1's direction, boots, and light. Ref: shot 1. |
| 3 | 2.0 s | Anant stops at the observatory threshold. | Anant | Half-buried sandstone doorway; twilight, crosswind. | Rear medium-wide; static. | Cut, 0.0 s. | Continue the approach; keep the moons behind him. Ref: shot 2. |
| 4 | 2.0 s | His gloved hand brushes salt from the astrolabe. | Anant, hand only | Observatory lintel; twilight spill, sheltered dust. | Close insert; locked. | Cut, 0.0 s. | Introduce one fixed brass astrolabe. Ref: shot 3. |
| 5 | 2.0 s | Anant turns the astrolabe's outer ring a quarter turn. | Anant, hands only | Same lintel; indigo shadow and copper highlights. | Macro; locked on the ring. | Cut, 0.0 s. | Continue the hand movement; preserve the ring design. Ref: shot 4. |
| 6 | 2.0 s | He makes one small adjustment to align the astrolabe with the moons. | Anant | Observatory roof opening; blue twilight, still interior. | Over-the-shoulder; gentle push-in. | Cut, 0.0 s. | Same hood, scarf, and fixed astrolabe. Ref: shot 5. |
| 7 | 2.0 s | Anant reaches toward the unlit sapphire prism. | Anant, hand only | Inner sandstone niche; cool shadows, settled dust. | Close insert; short push-in. | Cut, 0.0 s. | Establish the single prism and its gold filaments. Ref: shot 6. |
| 8 | 2.0 s | He lifts the prism from the niche. | Anant | Same niche; indigo interior, faint copper edge light. | Medium close-up; follow the lift. | Cut, 0.0 s. | Same prism; it remains with Anant from now on. Ref: shot 7. |
| 9 | 2.0 s | The prism emits one soft cyan-and-gold pulse. | Anant, hands only | Same alcove; dust catches the restrained glow. | Close insert; locked. | Cut, 0.0 s. | Preserve prism size, shape, and internal filaments. Ref: shot 8. |
| 10 | 2.0 s | A single star-map line points toward the red ridge. | Anant | Observatory doorway; late twilight, faint exterior dust. | Over-the-shoulder wide; tilt from prism to ridge. | Fade, 0.5 s. | Discovery beat; ridge direction matches shot 1. Ref: shot 9. |
| 11 | 2.0 s | Anant takes one step out toward the ridge. | Anant | Observatory threshold and red ridge; late twilight, dry wind. | Wide rear; gentle forward track. | Cut, 0.0 s. | Continue the projected route; prism stays with him. Ref: shot 10. |
| 12 | 2.0 s | He takes one deliberate step along the sandstone trail. | Anant, lower body | Ridge trail above the basin; deepening blue twilight, light dust. | Low side-track; left-to-right. | Cut, 0.0 s. | Keep travel direction and boots consistent. Ref: shot 11. |
| 13 | 2.0 s | Anant pauses as one gust lifts red dust. | Anant | Same ridge trail; dry gust, no storm. | Medium-wide; locked, dust crosses foreground. | Cut, 0.0 s. | Brief environmental movement only. Ref: shot 12. |
| 14 | 2.0 s | He raises his gaze toward the obsidian arch. | Anant | Red ridge with arch on the far crest; indigo night. | Medium close-up; very slow push-in. | Cut, 0.0 s. | The arch is the destination indicated in shot 10. Ref: shot 13. |
| 15 | 2.0 s | Anant tightens his charcoal scarf once. | Anant | Wind-exposed ridge; cold dry air and faint dust. | Close side-profile; locked. | Cut, 0.0 s. | Prism remains in his left hand; wardrobe unchanged. Ref: shot 14. |
| 16 | 2.0 s | He steps onto the stone bridge. | Anant | Bridge over a blue-mist ravine; cold night, dry wind above. | Wide side view; short lateral track. | Cut, 0.0 s. | Bridge is stable stone, not rope; continue toward frame right. Ref: shot 15. |
| 17 | 2.0 s | His right hand rests on the bridge parapet. | Anant, hands and torso | Same bridge; cool moonlight, mist far below. | Medium insert; gentle lateral drift. | Cut, 0.0 s. | Prism stays in left hand; match bridge stone. Ref: shot 16. |
| 18 | 2.0 s | Anant stops beneath the obsidian arch. | Anant | Ridge crest; twin moons, weaker wind, mist behind. | Low-angle wide; tilt from bridge to arch. | Cut, 0.0 s. | Match the destination and bridge direction. Ref: shot 17. |
| 19 | 2.0 s | His gloved palm touches the arch's three-point carving. | Anant, hands only | Obsidian arch; sheltered night, matte black stone. | Close insert; locked. | Cut, 0.0 s. | Keep the mark simple and non-textual; prism in other hand. Ref: shot 18. |
| 20 | 2.0 s | A thin cyan-and-gold line lights along the arch. | Anant | Same threshold; still air, twin-moon rim light. | Over-the-shoulder; slow upward tilt. | Fade, 0.5 s. | Light responds to the prism and carving. Ref: shot 19. |
| 21 | 2.0 s | Anant steps into the circular underground observatory. | Anant | Chamber beyond the arch; still air, suspended dust. | Wide interior; short rear dolly. | Cut, 0.0 s. | Introduce one central basalt dial; keep props unchanged. Ref: shot 20. |
| 22 | 2.0 s | He places the sapphire prism in the dial's central socket. | Anant | Center of the same chamber; muted blue-gold light. | Medium shot; gentle push-in. | Cut, 0.0 s. | Prism is the same one carried since shot 8. Ref: shot 21. |
| 23 | 2.0 s | One carved stone ring turns until its notch aligns with the prism. | Anant, hand only | Central basalt dial; still air, indigo shadow. | Macro; locked on one moving ring. | Cut, 0.0 s. | Keep the dial geometry simple; prism stays seated and the separate brass astrolabe remains at the surface lintel. Ref: shot 22. |
| 24 | 2.0 s | A thin golden circle appears around Anant's boots. | Anant, lower body | Chamber floor below the dial; still air and dust. | Top-down; locked. | Cut, 0.0 s. | One light circle only; follows the alignment in shot 23. Ref: shot 23. |
| 25 | 2.0 s | Anant lifts his eyes toward the ceiling. | Anant | Beside the dial; night, soft cyan-gold light. | Close-up portrait; subtle push-in. | Cut, 0.0 s. | Preserve face and hood; reflect the circle from shot 24. Ref: shot 24. |
| 26 | 2.0 s | A narrow seam of dawn-colored light appears in the wall. | Anant | Sealed chamber wall; still air, illuminated dust. | Medium-wide; slow pan from Anant to seam. | Cut, 0.0 s. | Light seam only; no collapsing wall or complex portal. Ref: shot 25. |
| 27 | 2.0 s | Anant takes one step toward the seam. | Anant | Same chamber between dial and wall; warm floor light. | Rear medium; short forward track. | Cut, 0.0 s. | Dial and seated prism remain behind him. Ref: shot 26. |
| 28 | 2.0 s | He steps through the threshold onto the salt basin. | Anant | Same basin and observatory; pre-dawn haze, wind easing. | Wide rear; static. | Cut, 0.0 s. | Return to shot 1's geography; twin moons fade. Ref: shot 27. |
| 29 | 2.0 s | Anant lowers the prism as first sunlight reaches the basin. | Anant | Open salt basin; first dawn, calm golden haze. | Medium-wide profile; gentle lateral drift. | Cut, 0.0 s. | Prism glow stays restrained; dawn changes gradually. Ref: shot 28. |
| 30 | 2.0 s | He takes one final step toward the golden horizon. | Anant | Reflective basin facing dawn; observatory recedes behind. | Wide rear closing shot; slow pull-back. | Cut, 0.0 s; final shot. | Close the twilight-to-dawn arc with the same traveler and prism. Ref: shot 29. |

## Review notes

- The planned story uses only one character and four connected locations; most
  shots are cuts, with crossfades only at the two act breaks.
- Every shot has one primary action and a specific subject, setting, lighting,
  and camera instruction. Full CogVideoX prompts are in the JSON file.
- Previous-shot IDs support editorial continuity planning but do not provide
  image conditioning to the current text-to-video provider.
- This is a plan only. No batch manifest was created, no videos were generated,
  and existing POC/test media was left untouched.
