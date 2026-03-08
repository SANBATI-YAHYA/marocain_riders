# Morocco Motorcycle Travel Knowledge Base
## Version 1.0 | AI Recommendation System

> **Purpose:** This knowledge base powers an AI system that generates personalized motorcycle trip recommendations across Morocco. It is structured for RAG (Retrieval-Augmented Generation), vector embedding, and GPX-linked itinerary planning.

---

# SECTION 1: PLACES

## Places Table

| place_id | name | region | province | latitude | longitude | place_type | primary_vibe | secondary_vibes | budget_level | recommended_stay_duration | best_season | altitude_category | road_accessibility | bike_type_suitability | beginner_friendliness | scenic_score | comfort_score | fuel_access_score | available_activities | safety_notes | weather_notes |
|----------|------|--------|----------|----------|-----------|------------|--------------|-----------------|--------------|--------------------------|-------------|-------------------|--------------------|-----------------------|-----------------------|--------------|---------------|-------------------|----------------------|--------------|---------------|
| PL001 | Marrakech | Marrakech-Safi | Marrakech | 31.628 | -7.992 | city | urban_gateway | culture, food, commerce | medium | 1-2 days | Oct-Apr | low (460m) | excellent | all types | 5/5 | 9/10 | 8/10 | 9/10 | medina, souks, Jemaa el-Fna, bike shops | high traffic in medina, park outside old city | hot summers (40°C+), mild winters |
| PL002 | Ouarzazate | Draa-Tafilalet | Ouarzazate | 30.920 | -6.893 | town | desert_gateway | film, palmery, culture | low-medium | 1-2 days | Sep-May | medium (1135m) | excellent | all types | 4/5 | 7/10 | 8/10 | 7/10 | Aït Benhaddou, kasbah visits, cinema museum | safe, tourist-oriented | very hot Jul-Aug (42°C+), cold nights Dec-Jan |
| PL003 | Zagora | Draa-Tafilalet | Zagora | 30.329 | -5.838 | town | desert | oasis, dunes, palmery | low | 1 day | Oct-Apr | low (660m) | good | adventure, dual-sport | 3/5 | 7/10 | 7/10 | 6/10 | camel treks, palmery rides, Draa Valley | safe but remote, heat risk Jun-Sep | extreme heat summer, sand hazard spring |
| PL004 | Merzouga | Draa-Tafilalet | Errachidia | 31.101 | -4.013 | village | desert | dunes, erg, bivouac | low-medium | 1-2 nights | Oct-Apr | low (1030m) | moderate (last 50km sandy) | adventure, dual-sport | 3/5 | 8/10 | 6/10 | 5/10 | Erg Chebbi dunes, sunset rides, quad/camel | sand road risk, no asphalt to dunes | extreme heat summer, cold winter nights, sandstorms spring |
| PL005 | Todra Gorge | Draa-Tafilalet | Tinghir | 31.591 | -5.600 | natural_site | gorge | dramatic, canyon, climbing | low | 0.5-1 day | Mar-Nov | medium (1400m) | good | all types | 5/5 | 6/10 | 5/10 | 5/10 | canyon ride-through, rock climbing, lunch stop | narrow road, tourist crowds midday | flash flood risk after rain, cold shadow zones |
| PL006 | Dades Gorge | Draa-Tafilalet | Tinghir | 31.523 | -6.017 | natural_site | gorge | dramatic, winding road, kasbah | low | 1 night | Apr-Oct | medium (1500m) | good | all types | 5/5 | 6/10 | 5/10 | 4/10 | famous hairpin bends, kasbah photography, village walks | road narrows upper section | flash floods, some sections steep |
| PL007 | Tizi n'Tichka Pass | Marrakech-Safi / Draa-Tafilalet | Marrakech / Ouarzazate | 31.253 | -7.374 | mountain_pass | mountain | dramatic, high altitude, views | low | stop only | Mar-Nov | high (2260m) | good paved | adventure, touring, dual-sport | 3/5 | 9/10 | 3/10 | 3/10 | summit photos, tea stop, Atlas views | fog, ice Oct-Apr, sharp turns | snow possible Nov-Mar, strong winds, cold |
| PL008 | Aït Benhaddou | Draa-Tafilalet | Ouarzazate | 31.047 | -7.130 | heritage_site | culture | UNESCO, ksar, film sets | low | half-day | Sep-May | medium (1000m) | good | all types | 4/5 | 5/10 | 6/10 | 3/10 | ksar visit, film location walk | very touristy midday | dusty in summer, can be cold in winter |
| PL009 | Midelt | Draa-Tafilalet | Midelt | 32.685 | -4.733 | town | transition | apple orchards, Atlas, cool air | low | 0.5-1 day | Apr-Oct | high (1508m) | excellent | all types | 4/5 | 6/10 | 6/10 | 7/10 | town lunch stop, apple market, Atlas viewpoints | safe, calm, underrated stop | cold winters, pleasant spring/autumn |
| PL010 | Erfoud | Draa-Tafilalet | Errachidia | 31.434 | -4.235 | town | desert_town | fossil market, palmery | low | 0.5 day | Oct-Apr | low (811m) | excellent | all types | 3/5 | 6/10 | 7/10 | 7/10 | fossil shops, date palms, Merzouga gateway | safe, good resupply point | hot summer, sandstorms possible |
| PL011 | Tinerhir | Draa-Tafilalet | Tinghir | 31.515 | -5.524 | town | valley | oasis, palmery, Todra gateway | low | 0.5-1 day | Apr-Oct | medium (1342m) | excellent | all types | 4/5 | 6/10 | 7/10 | 7/10 | palmery walk, kasbah views, refueling | safe, reliable fuel | hot summers, pleasant spring/autumn |
| PL012 | Fès | Fès-Meknès | Fès | 34.037 | 5.000 | city | culture | medina, imperial, crafts | medium-high | 1-2 days | Mar-May, Sep-Nov | low (410m) | excellent | all types | 3/5 | 8/10 | 9/10 | 9/10 | medina walk, tanneries, Bou Inania | heavy traffic around medina | hot summers, cold winters, mild spring/autumn |
| PL013 | Chefchaouen | Tanger-Tetouan-Al Hoceima | Chefchaouen | 35.168 | -5.268 | mountain_town | scenic | blue city, Rif, hiking | low-medium | 1 day | Apr-Jun, Sep-Oct | medium (600m) | good | all types | 4/5 | 8/10 | 7/10 | 5/10 | old medina, Ras el-Maa, Rif hikes | light traffic, mellow vibe | rainy winters, fog possible, comfortable summers |
| PL014 | Tizi n'Test Pass | Marrakech-Safi / Souss-Massa | Taroudant | 30.870 | -8.503 | mountain_pass | adventure | remote, wild, epic views | low | stop only | Apr-Oct | high (2092m) | winding paved | adventure, dual-sport | 2/5 | 9/10 | 2/10 | 2/10 | summit views, Tin Mal mosque | very isolated, limited phone | snow possible Dec-Feb, strong winds, fog |
| PL015 | Taroudant | Souss-Massa | Taroudant | 30.472 | -8.876 | town | authentic | souks, rose-colored walls, calm | low | 0.5-1 day | Nov-Apr | low (270m) | excellent | all types | 4/5 | 7/10 | 7/10 | 8/10 | medina, ramparts, souks, lunch stop | very safe and relaxed | hot summers, pleasant winters |
| PL016 | Agadir | Souss-Massa | Agadir | 30.428 | -9.598 | city | coastal | beach, resort, services | medium-high | 1 day | year-round | low (sea level) | excellent | all types | 4/5 | 6/10 | 9/10 | 10/10 | beach, corniche, moto service centers | busy city, bike theft risk | mild all year, Atlantic winds, fog mornings |
| PL017 | Skoura | Draa-Tafilalet | Ouarzazate | 31.063 | -6.559 | village | oasis | rose valley, palmeraie, quiet | low | 1 night | Oct-May | medium (1100m) | good | all types | 4/5 | 7/10 | 6/10 | 4/10 | palmery walks, kasbah visits, rose harvest May | very relaxed, off main road | hot summers, pleasant spring |
| PL018 | Boumalne Dadès | Draa-Tafilalet | Tinghir | 31.371 | -5.978 | town | valley | Dades gorge gateway, views | low | 0.5-1 day | Apr-Oct | medium (1500m) | excellent | all types | 4/5 | 6/10 | 6/10 | 7/10 | Dades gorge entry, valley views, lunch | safe, calm | cold winters, pleasant spring/autumn |
| PL019 | Rissani | Draa-Tafilalet | Errachidia | 31.280 | -4.260 | town | desert_town | market, history, Merzouga gateway | low | 0.5 day | Oct-Apr | low (930m) | good | all types | 3/5 | 5/10 | 6/10 | 6/10 | market day, kasbah ruins, Merzouga access | piste track begins near here | hot summer, cold winter nights |
| PL020 | Msemrir | Draa-Tafilalet | Tinghir | 31.691 | -5.820 | village | remote | high valley, Berber, isolated | very low | 1 night | May-Oct | high (1800m) | piste (challenging) | adventure, dual-sport | 1/5 | 9/10 | 3/10 | 3/10 | high valley exploration, M'Goun trekker base | very remote, limited services | snow Oct-Apr, cold nights, flash flood risk |

---

## Place Descriptions

### PL001 — Marrakech
Marrakech is the primary gateway city for motorcycle trips heading south through the High Atlas or southwest toward Agadir and the Atlantic coast. For riders, the city itself is a controlled chaos of scooters, donkey carts, and tourist taxis — thrilling to navigate but frustrating if you lack local traffic sense. Riding into the medina is inadvisable; the souks surrounding Jemaa el-Fna are impassable on a large adventure bike. Most riders park on the ring roads or at their riad's indicated parking and explore on foot.

Arriving from the north on the A7 motorway gives a smooth approach. From the east via Tizi n'Tichka, the descent into Marrakech through the suburbs is rewarding after the drama of the pass. From the south on the N9, the palmery and irrigated gardens of the Haouz Plain announce your arrival gently.

The city is an ideal start-of-trip supply point — fuel, food, spare chains, tire repairs, camping gear, and pharmacies are all accessible. Budget travelers can find riads with courtyard parking from €25/night; mid-range options with secure garages run €50–80/night. The Hivernage district outside the medina walls offers quiet streets and reliable restaurant options without the chaos.

Weather-wise, Marrakech is best avoided in July and August (38–42°C). Spring (March–May) and autumn (September–November) offer ideal conditions: 22–28°C with clear skies. The city is a poor overnight choice for riders who simply want to cross the Atlas — it adds city riding stress and cost. If you must stay, aim for the northern medina edge near Bab Doukkala where parking is somewhat easier.

---

### PL002 — Ouarzazate
Known internationally as the "Gateway to the Sahara" and Morocco's film capital, Ouarzazate is a critical junction town for nearly every southern Morocco motorcycle circuit. At 1135m elevation, it offers noticeably cooler temperatures than the lower desert basins, and its commercial boulevard (Avenue Mohammed V) provides reliable fuel stations, ATMs, pharmacies, and several mechanics capable of working on larger bikes.

Arriving from Marrakech over Tizi n'Tichka, the descent into Ouarzazate feels like a reward — the brown-gold landscape opens up, the High Atlas recedes behind you, and the road straightens through a plateau before dropping into the town. From the east via the N10 (Road of a Thousand Kasbahs), Ouarzazate is a logical refuel and rest point before or after the dramatic sections further east.

Riders of all experience levels are comfortable here. Road quality on all approaches is good paved. The kasbah of Taourirt in town and the nearby Aït Benhaddou ksar (18km away) are easy half-day detours that require no off-pavement riding. Budget guesthouses along the main avenue start at €20/night with basic motorcycle parking. Mid-range hotels with locked garages run €50–70. The Berbère Palace and similar properties cater to tour groups but often have guarded parking useful for night stops.

Avoid town as a base in July and August unless your bike is in for repair — the heat is punishing and there's no shade riding to speak of. Nights in December–January can drop to 2–4°C; pack accordingly if transiting the south in winter.

---

### PL003 — Zagora
Zagora sits at the far southern end of the Draa Valley, surrounded by date palm oases and pointing the way toward the Sahara with its famous "Timbuktu 52 Days" sign. For motorcycle travelers, it functions as either a pleasant overnight destination in a palmery guesthouse or a supply/fuel checkpoint before heading further into the desert toward M'Hamid el-Ghizlane and the Erg Chigaga dunes.

The N9 south from Ouarzazate runs down the entirety of the Draa Valley — 165km of increasingly remote, increasingly arid, increasingly beautiful road. The pavement is largely good quality though patchy in sections. The valley walls close in and widen alternately, exposing ancient kasbahs and ksar villages along both banks of the Draa river. This is one of Morocco's most photogenic road sequences for motorcyclists.

Zagora itself is a modest town. Its services include two gas stations, several basic supermarkets, a pharmacy, and a handful of guesthouses ranging from dirt-cheap rooftop rooms (€10) to pleasant palmery auberges with pools (€45–60). The town is straightforward to navigate on a bike. Fuel up reliably in Zagora before pushing further south — the next reliable station is M'Hamid 100km south, and it is not always stocked.

Summer heat in Zagora is severe: 42–46°C in July–August is not unusual. Dust and sand haze from the Sahara can reduce visibility to 200–400m during spring khamsins. The best riding season is October through April, when temperatures stay in the 22–32°C range during the day.

---

### PL004 — Merzouga
Merzouga is the most famous desert stop in Morocco, set beside the immense Erg Chebbi — a sea of golden dunes reaching 150m high. For motorcycle travelers, it is a must-see landmark, but also a logistical challenge: the last 15–20km from Rissani involves either the paved N13 (acceptable) or various sandy pistes that cut corners (for experienced adventure riders only).

The iconic image of a loaded adventure bike parked before the Erg Chebbi dunes at sunset is a Morocco motorcycling staple, and for good reason — the spectacle is genuinely extraordinary. Riders with dual-sport bikes can explore sandy tracks at the dune base; road-bikers should stay on the N13 and park in Merzouga village.

Guesthouses cluster densely near the dune edge. Prices range from €15 for basic rooms with shared facilities to €90+ for air-conditioned auberges with pools and terraces facing the dunes. Most have sand-floor parking areas — not ideal for heavy bikes on center stands, but manageable. Book ahead in October–November and February–March (peak desert season). The town has a fuel station but it is not always open or stocked — fill in Erfoud 53km west. Basic tagine restaurants are abundant and cheap (€5–8 per meal).

Cell coverage is weak to nonexistent at the dune edge; mobile data requires positioning toward the village center. Riders with GPS should download offline maps before arrival. In summer, Merzouga exceeds 48°C and is genuinely dangerous for riding through the heat of the day.

---

### PL005 — Todra Gorge
The Todra Gorge is one of Morocco's most spectacular natural formations — a slot canyon where vertical limestone walls rise 300m either side of a narrow road and shallow river. For motorcycle travelers, it is a perfect stop, combining dramatic scenery, a manageable approach road, and a handful of basic cafés and auberges tucked against the cliff base.

The approach from Tinerhir (14km) follows the Todra riverbed, climbing gently through a green palmery before the gorge walls close in dramatically. The road is paved but narrow in the gorge itself — two vehicles can pass, but only just. Large adventure bikes need careful positioning when tourist buses are present (typically 10am–2pm). Early morning or late afternoon riding gives you the gorge largely to yourself, with extraordinary light on the canyon walls.

The gorge is a natural turnaround point for road bikes — the paved road ends in the lower gorge and transitions to rough piste beyond. For adventure bikes and experienced dual-sport riders, continuing north through the gorge and up to the plateau toward Msemrir (PL020) is a spectacular and demanding full-day route. This segment requires real off-road capability and ideally a GPS track.

Budget is minimal here: a roadside tagine lunch runs €5–7, and simple overnight rooms at cliff-base auberges go for €15–25. Cold water is the norm — hot showers are rare. The gorge is a rest stop, not an overnight base, for most riders. Weather risk: flash floods can occur with zero local warning when rain falls on distant High Atlas ridges. If the sky darkens upstream, leave the gorge immediately.

---

### PL006 — Dades Gorge
The Dades Gorge is renowned among motorcyclists worldwide for one specific section: the series of extreme hairpin bends at its upper end, where the road twists back on itself six times in under a kilometer across a sheer red rock face. This section, often called the "Monkey Fingers," is photographed constantly from above, and for good reason — it is a deeply satisfying piece of riding.

The approach from Boumalne Dadès (PL018) climbs through a broad valley of rose gardens and kasbah villages before narrowing into the gorge. The lower gorge section is wide and pleasant; the middle section passes auberges and rose bushes; the upper hairpins are the climax. Road quality is generally good on the lower section, deteriorating slightly on the upper bends, which can have loose gravel in corners after rain.

Overnight in the gorge rather than rushing back to Boumalne — several auberges at the 20–25km mark offer clean rooms (€20–35), hot showers, and tagine dinners with gorge views. The atmosphere is dramatically different from the busy Ouarzazate–Tinerhir corridor: quiet, fresh air, no tourist buses past the lower section, and genuine Berber village hospitality.

Adventure riders can push further up the gorge and eventually link to the high plateau, connecting with the Msemrir (PL020) sector — a remote, demanding piste route that requires significant off-road experience. In spring (April–May), the rose harvest transforms the lower valley into a fragrant, photogenic experience. Flash flood risk increases with High Atlas snowmelt in March–April.

---

### PL007 — Tizi n'Tichka Pass
At 2260m, Tizi n'Tichka is the highest paved road pass in Morocco and the primary route connecting Marrakech to the south. For motorcycle travelers, crossing it is a rite of passage — the road climbs from the Haouz Plain in a series of long, well-graded switchbacks through Berber villages of stone and ochre, cresting into a moonscape of bare rock and snow in winter, then descending steeply toward the Draa-Tafilalet region.

The road quality on the N9 over Tichka is generally good to excellent, having been repeatedly repaired. However, surface conditions deteriorate at the pass itself — patches, frost damage in winter, and the occasional poorly repaired section require attention. Guardrails are present on most critical sections but not everywhere. Blind corners are common; horn use is culturally expected and practically necessary.

The pass is not suitable for novice riders in wet or cold conditions. In dry spring and autumn weather, it is an accessible and deeply rewarding ride — the kind of route that makes riders stop repeatedly just to absorb the panorama. The summit area has tea sellers, argan oil stalls, and amethyst vendors. Cell coverage drops in mid-mountain sections.

Snow closes or severely restricts the pass typically from late November through mid-March. Summer crossings are fine but watch for overheating (engine and rider) on the climb from Marrakech, especially loaded touring bikes. Wind can be significant on the exposed summit sections. A fuel stop before leaving Marrakech is essential — no fuel between the city outskirts and Aït Ourir (30km) or between the upper pass and the N9 junction 40km south.

---

### PL008 — Aït Benhaddou
Aït Benhaddou is the most photographed ksar (fortified village) in Morocco and a UNESCO World Heritage site, serving as a film backdrop for dozens of major productions. For motorcycle travelers, it is a half-day excursion from Ouarzazate (18km north on a good paved road). The ride itself is pleasant but not exceptional — it's the destination that justifies the stop.

The visual impact of the ksar rising from the Ounila riverbed, all stacked brown earth towers and crenellated walls, is genuinely impressive. Entry requires a wade or stepping-stone crossing of the shallow river (hilariously awkward in full motorcycle gear). The village is heavily touristic midday but calm in early morning and late afternoon.

Riders staying in Ouarzazate should budget 3–4 hours for the roundtrip with a walk inside the ksar. There's a small cluster of café-restaurants at the foot of the ksar serving decent tagines (€6–9) and the obligatory fresh-squeezed orange juice. Parking for motorcycles is easy and unproblematic. Not a recommended overnight stop — Ouarzazate provides better facilities and is only 18km away.

---

### PL009 — Midelt
Midelt is the classic halfway point between Fès and Merzouga, sitting in a broad valley between the Middle Atlas and the Anti-Atlas at 1508m elevation. It is an underrated gem for motorcycle travelers — a friendly, functional market town with reliable services, excellent lamb tagine, and a distinctly cooler, fresher atmosphere compared to the desert towns further south.

The N13 approaching Midelt from the north drops through a spectacular canyon section (Gorges du Ziz, see below) before opening into the Midelt basin — a visual transition from green mountain to golden plain that never gets old. From the south, the approach is across a barren plateau, which makes Midelt's orchards and green edges feel like a genuine oasis arrival.

Midelt is a working Moroccan town untouched by mass tourism, which means authentic café culture, cheap meals (€4–7), and genuine interactions. The Monday and Thursday markets are worth timing a stop around. Fuel is reliable at multiple stations on the main N13 axis. Basic hotels run €15–25; the town has no luxury options, which suits most riders perfectly.

For riders doing the full Fès–Merzouga route in two days, Midelt makes a logical overnight stop (approximately midpoint). The region's autumn apple harvest (September–October) adds a scenic bonus. Cold and wet winters (occasional snow) make the town a less appealing winter stop.

---

### PL010 — Erfoud
Erfoud serves as the last reliable resupply point before the Erg Chebbi desert zone, and as such is an important logistical stop for motorcycle travelers heading to or from Merzouga. The town is known for its fossil market (Devonian-era trilobites and sea lilies are extracted from the surrounding Paleozoic limestone) and its date festival in October.

For riders, Erfoud's primary value is practical: fuel stations, a pharmacy, a basic mechanic's street (central market area), a decent supermarket, and several guesthouses that cater to adventure bikers passing through. It is 53km west of Merzouga — a comfortable leg that can be timed before or after the dune visit.

The town is not particularly scenic but is genuinely pleasant in the October date-harvest season when market activity peaks. Budget hotels start at €15; mid-range options with air conditioning (essential in summer if you must transit) run €35–50. The main street tagine restaurants serve reliable, cheap food. Fill up at Erfoud before heading east — the Merzouga station is not always open or stocked.

---

### PL011 — Tinerhir
Tinerhir (also spelled Tinghir) is the provincial capital that acts as the gateway town to the Todra Gorge, sitting in a wide and visually spectacular palmery valley. For motorcycle travelers, it is a natural fuel, food, and rest stop on the R702/N10 corridor — the famous "Road of a Thousand Kasbahs."

Arriving from either direction on the N10, the palmery views from the road above Tinerhir are outstanding — a dense green strip of date palms backed by red-ochre cliffs. The town center is easy to navigate on a bike; fuel and ATM access are reliable. Basic guesthouses (€15–25) and a few mid-range options (€40–60) serve all budget levels.

The Todra Gorge entrance is 14km north; most riders use Tinerhir as a lunch stop and continue to the gorge as a late-afternoon excursion before returning for overnight. Alternatively, overnight at the gorge itself for a more atmospheric experience.

---

### PL012 — Fès
Fès is Morocco's spiritual and intellectual capital, home to the world's oldest continuously operating university and a medieval medina of extraordinary complexity. For motorcycle travelers, Fès is primarily a start or end point for northern routes — the Rif mountains, the Middle Atlas, or the trans-Atlas crossing to Merzouga.

Riding into Fès is manageable but demanding — the ring road (Route de Meknès, Route d'Immouzer) allows clean access to medina-adjacent hotels without entering the labyrinthine old city streets. Parking is available at several designated zones near Bab Bou Jeloud. A guarded parking area specifically for motorcycles and bicycles operates near the main medina gates.

Services for riders are excellent: fuel stations on all approach roads, mechanics in the Ville Nouvelle (New Town), tire repair shops near the bus station area, and comprehensive shopping for supplies. Mid-range hotels in the Ville Nouvelle with secure parking run €50–80. Riad accommodations in the medina are memorable but require parking elsewhere.

Fès is best visited in spring or autumn; summers are hot (38°C+) and winters bring real rain and occasional cold snaps. The medina itself rewards 6–8 hours of walking exploration — budget at least a full day if cultural immersion is a goal.

---

### PL013 — Chefchaouen
Chefchaouen, the "Blue City" of the Rif Mountains, is one of Morocco's most photographed and most visited small towns. For motorcycle travelers approaching from Fès or the coast, it combines genuine scenic appeal (blue-painted streets climbing the hillside, mountain backdrop, fresh Rif air) with manageable access roads and a relaxed, tolerant vibe.

The ride up from Tetouan or from Ouazzane takes riders through the Rif foothills — green, forested, and relatively empty of traffic. The approach from the east via Ketama and the N2 is more dramatic (high Rif plateau) but involves occasional road quality challenges. Chefchaouen sits at 600m and offers genuinely cool, fresh air compared to the coastal plains.

Parking for bikes is easy at several dedicated areas outside the medina walls. The medina itself is walkable in 2–3 hours. Guesthouses in the medina offer rooftop terraces and mountain views (€25–50). The café culture is relaxed and excellent — proper espresso and Rif honey are highlights. The town is almost entirely pedestrian inside the walls; motorbike noise is socially frowned upon near the main squares.

Chefchaouen fits naturally into a northern Morocco circuit (Tangier–Chefchaouen–Fès–Ifrane–Azrou), providing a contrast between coastal, mountain, and imperial city experiences.

---

### PL014 — Tizi n'Test Pass
Tizi n'Test (2092m) is the wilder, lonelier, and arguably more rewarding alternative to Tizi n'Tichka. The road connecting Marrakech to Taroudant via the R203 is narrower, less traveled, and more dramatic — climbing through the High Atlas with sheer drops, a narrow asphalt ribbon, no guardrails on many sections, and views that can include the Toubkal massif on clear days.

This is not a road for beginners or large touring bikes in poor condition. The road surface is paved throughout but with sections of serious patching, occasional rockfall debris, and extreme camber changes. The legendary section near the summit involves single-lane road carved into the cliff face. Traffic is minimal — which is both the magic and the danger. Cell coverage is essentially nonexistent for 80km.

The Tin Mal mosque (12th-century Almohad dynasty) sits 8km below the summit on the southern descent — a stunning medieval ruin in a mountain valley that rewards the effort of reaching it. Arriving at Taroudant after the descent feels enormously satisfying: the transition from wild mountain to orange-tree-lined streets and rose-colored ramparts is one of Morocco's great riding arrivals.

Fuel must be planned: fill up in Marrakech (or Asni) before the pass; the first reliable station on the south side is at Ouled Berhil, 25km past Taroudant. Do not attempt in wet or icy conditions. Best ridden in April–October. An early morning start from Marrakech allows reaching Taroudant by early afternoon, maximizing light and avoiding afternoon heat.

---

### PL015 — Taroudant
Taroudant is often called "Little Marrakech" — a walled city surrounded by almond and argan orchards in the Souss Valley, with none of Marrakech's mass tourism and all of its essential charm. For motorcycle travelers, it is a wonderful overnight destination that rewards slow exploration: the ramparts can be circumnavigated by bike in 20 minutes, and the souk offers genuinely local goods rather than tourist merchandise.

The approach from any direction is pleasant: from the north via Tizi n'Test (dramatic), from the east via Aït Melloul and the N10 (comfortable), or from the west via Agadir (fast, flat, 80km on good road). The town center is navigable on a bike; parking outside the walls is straightforward. The main square (Arsat el-Massira and Place Assarag) has café terraces perfect for post-ride relaxation.

Budget accommodation (€20–35) and mid-range riads (€50–80) both offer pleasant options; the riads inside the medina walls have parking arrangements with nearby guarded lots. Restaurants in the souk area serve excellent harira, lamb tagine with prunes, and fresh-squeezed orange juice at very reasonable prices (€5–10 per meal).

Taroudant is at its best November through April — temperatures in the 18–25°C range, clear skies, and the almond blossom (February) or orange harvest (November–December) providing exceptional scenery. Summer (June–September) brings 38–42°C heat and most accommodation shifts to evening-only activity.

---

### PL016 — Agadir
Agadir is Morocco's main beach resort city and the country's best-developed coastal urban center south of Casablanca. For motorcycle travelers, it functions primarily as a logistics hub rather than a destination in its own right: excellent fuel access, major workshops (including brands like Michelin tire dealers), ATMs, supermarkets, ferry connections (historically), and airport links for trip starts and ends.

The city was rebuilt entirely after a 1960 earthquake and lacks the historic fabric of other Moroccan cities. The seafront corniche is pleasant for an evening ride but uninspiring. The primary appeal for riders is reliability: anything your bike needs can be sourced here. Major brand mechanics (Honda, Kawasaki dealers) operate in the industrial zone south of the city.

Accommodation ranges from backpacker hostels (€15) to international beach hotels (€120+). For riders doing the Atlantic coast route or the southern Anti-Atlas loop, Agadir is either a start/end point or a mid-route service day. Bike security: avoid leaving bikes unattended in exposed public areas overnight — use hotel parking.

---

### PL017 — Skoura
Skoura is a large oasis of date palms and ancient kasbahs, located between Ouarzazate and Boumalne Dadès on the N10. It is an off-the-radar gem for motorcycle travelers — most riders blast through on the way to the Dades and Todra gorges without stopping, missing one of Morocco's most photogenic and peaceful spots.

The oasis is threaded by sandy pistes navigable on any adventure bike and on foot — a 90-minute exploration through kasbah ruins and irrigation channels is highly recommended. The famous Amerhidil Kasbah stands at the edge of the palmery and opens for guided visits. In May, the rose harvest in this area produces extraordinary fragrance and color.

Several auberges within the palmery (€25–50) offer rooms opening directly onto rose gardens or date palms — among the most atmospheric overnight stops in southern Morocco. Dinner in these guesthouses (€10–15 full menu) typically involves fresh vegetables and slow-cooked tagine made on-premises. Fuel is not available in Skoura — use Ouarzazate or Boumalne Dadès.

---

### PL018 — Boumalne Dadès
Boumalne Dadès is the town at the entrance of the Dadès Gorge and a reliable service stop on the N10. It sits on a prominent ridge with views down the Dadès Valley in both directions — a genuinely photogenic town setting that most riders miss because they're looking at the road. The town has fuel, basic supermarkets, a Saturday market, and several guesthouses.

For riders, Boumalne is a practical overnight base for exploring the Dadès Gorge the following morning in ideal light conditions. It is also a natural lunch stop between Ouarzazate (100km west) and Tinerhir (60km east). The N10 through here is smooth and fast — probably the best quality paved road in the entire southern circuit. Accommodation €15–40, fuel reliable.

---

### PL019 — Rissani
Rissani is a historic oasis town that was the birthplace of the Alaouite dynasty (Morocco's ruling family). For motorcycle travelers, it is the last proper town before Merzouga (40km southeast) and the start of the sandy piste network that extends into the Erg Chebbi. It hosts a colorful three-days-per-week souk (Sunday, Tuesday, Thursday) that draws Saharan traders from surrounding villages.

Fuel availability in Rissani is good on market days but can be limited otherwise — always top up here before Merzouga. Basic mechanical help (punctures, chain adjustments) is available from workshop stalls near the market. The town itself is dusty and functional rather than scenic, but the souk atmosphere is authentic and engaging.

---

### PL020 — Msemrir
Msemrir is a small Berber village in the high Dadès Valley, accessible only by piste from either the Todra Gorge plateau road or the upper Dadès Gorge. At 1800m elevation, it is remote, beautiful, and completely off the tourist trail. For experienced adventure and dual-sport riders, it represents the soul of Moroccan mountain motorcycling — total isolation, dramatic landscape, authentic village hospitality.

The piste from Boumalne Dadès through the upper gorge to Msemrir is approximately 60km of challenging dirt track: river crossings, rocky climbs, loose shale descents, and complete absence of phone signal or rescue infrastructure. The same is true approaching from the Todra Gorge plateau. A GPS track (GPX file) and companion rider are strongly recommended.

Msemrir has a single auberge-style guesthouse offering basic rooms (€10–15) and home-cooked meals. Fuel is not available — riders must carry sufficient range for a 120–150km offroad segment. In winter, the village can be snowbound and completely inaccessible. This is a route for experienced riders with appropriate adventure bikes (≥21" front wheel, good ground clearance), solid off-road skills, and proper preparation.

---

# SECTION 2: ROUTES

## Routes Table

| route_id | route_name | linked_gpx_name | start_place_id | end_place_id | via_place_ids | region | route_type | distance_km | estimated_ride_hours | recommended_days | difficulty_level | bike_type_suitability | beginner_friendliness | road_surface_summary | elevation_profile | scenic_features | hazard_level | fuel_gap_km | cell_coverage_level | best_season | avoid_season | weather_dangers | recommended_stay_ids | recommended_restaurant_ids | recommended_fuel_station_ids | route_summary | rider_advice | safety_advice | why_choose_this_route |
|----------|------------|-----------------|---------------|-------------|---------------|--------|------------|-------------|----------------------|-----------------|-----------------|----------------------|----------------------|----------------------|-------------------|-----------------|--------------|-------------|--------------------|-----------|--------------|-----------------|--------------------|--------------------------|---------------------------|---------------|-------------|--------------|----------------------|
| RT001 | Grand Atlas Circuit | grand_atlas_circuit.gpx | PL001 | PL001 | PL007,PL002,PL008,PL017,PL006,PL005,PL011,PL010,PL004,PL019,PL009 | Multi-region | loop_circuit | 1450 | 16-20 | 7-10 | intermediate | all-road touring, adventure | 3/5 | 85% paved, 15% compacted gravel near Merzouga | undulating with two major passes (2260m, 1500m plateau) | Tizi n'Tichka, Draa Valley, Dades Gorge, Todra Gorge, Erg Chebbi | medium | 90 (Erfoud-Merzouga) | medium | Oct-Apr | Jul-Aug | heat, sand, mountain cold | ST001,ST002,ST004,ST005,ST006,ST007 | RS001,RS002,RS003,RS004 | GS001,GS002,GS003,GS004 | Classic 7-10 day loop from Marrakech over Atlas south to desert and back | Never skip Erfoud fuel fill; use dawn starts in summer; book Merzouga ahead Oct-Nov | Check Tichka pass weather before departure in Nov-Mar | The complete Morocco south experience: mountains, valleys, gorges, desert |
| RT002 | Road of a Thousand Kasbahs | kasbahs_road.gpx | PL002 | PL011 | PL008,PL017,PL018,PL006,PL005 | Draa-Tafilalet | linear_scenic | 320 | 5-6 | 2 | easy-intermediate | all types | 4/5 | 90% excellent paved | gently undulating, 1000-1500m plateau | kasbah villages, palmeries, gorge entries, rose valleys | low | 110 (Ouarzazate-Skoura-Boumalne) | medium | Apr-Oct | Jul-Aug heat | heat, rare flash flood | ST002,ST003,ST007,ST008 | RS002,RS004,RS005 | GS002,GS005 | Morocco's most beautiful paved road, threading kasbahs and gorge gateways | Start before 8am from Ouarzazate to reach gorges in good light | Watch for trucks on N10 blind bends | Best pure culture-and-scenery ride for all skill levels |
| RT003 | Tizi n'Test Adventure | tizi_test_adventure.gpx | PL001 | PL016 | PL014,PL015 | Marrakech-Safi / Souss-Massa | linear_mountain | 300 | 5-6 | 1-2 | advanced | adventure, dual-sport, capable touring | 2/5 | 100% paved, narrow, patched | major climb to 2092m then long southern descent | Tin Mal mosque, Atlas cliffs, Sous Valley panorama | high | 95 (Marrakech-Taroudant) | low (40km zero signal) | Apr-Oct | Nov-Mar | mountain cold, fog, ice, rockfall | ST009,ST010 | RS006,RS007 | GS006,GS007 | Hardest paved pass in Morocco, wildly beautiful, extremely isolated | Fill tank in Marrakech; inform accommodation of ETA; bring warm layer for summit | No guardrails in key sections; no rescue for 80km stretch | The ultimate Morocco mountain pass experience for confident riders |
| RT004 | Sahara Desert Express | sahara_express.gpx | PL012 | PL004 | PL009,PL010,PL019 | Draa-Tafilalet | linear_desert | 580 | 8-10 | 2-3 | easy-intermediate | all types | 4/5 | 95% excellent paved N13 | long gradual descent from Midelt plateau to desert floor | Ziz Gorge, Tafilalet palmery, Erg Chebbi | low-medium | 90 (Erfoud-Merzouga) | medium-low | Oct-Apr | Jul-Aug | heat, sand, crosswinds | ST005,ST011,ST012 | RS003,RS008 | GS003,GS008 | Classic Fès-to-desert route, all-paved, great for first-timers | Book Merzouga guesthouse by midday; fuel at Erfoud | Ziz Gorge section has sharp tunnels — use lights | Best intro route for new Morocco riders: long, paved, manageable |
| RT005 | Atlantic Coast & Anti-Atlas | atlantic_antiatlas.gpx | PL016 | PL001 | PL015,PL014 | Souss-Massa / Marrakech-Safi | linear_coastal_mountain | 420 | 7-8 | 2-3 | intermediate-advanced | adventure, dual-sport, capable touring | 3/5 | 80% good paved, 20% mountain road | coastal plain then dramatic climb over Tizi n'Test | Atlantic coast, Souss Valley, Anti-Atlas, High Atlas | medium-high | 95 | low in mountains | Apr-Jun, Sep-Oct | Jul-Aug, Nov-Mar | mountain cold, Atlantic fog, pass ice | ST009,ST010,ST013 | RS006,RS007,RS009 | GS006,GS007,GS009 | Combines coastal Morocco with the epic Tizi n'Test crossing | Plan Taroudant overnight between coast and mountain | Check Test pass conditions before leaving Taroudant | The complete south-western Morocco experience: sea, valley, and mountain |
| RT006 | Rif Mountains Loop | rif_loop.gpx | PL012 | PL012 | PL013 | Tanger-Tetouan-Al Hoceima | loop_circuit | 480 | 8-10 | 3-4 | intermediate | adventure, touring, dual-sport | 3/5 | 75% good-excellent paved, 25% winding mountain | significant elevation changes, 600-1600m Rif plateau | Blue City, Rif forests, Mediterranean viewpoints | medium | 80 | medium-low in Rif | Apr-Jun, Sep-Oct | Dec-Mar | mountain rain, fog, wet forest roads | ST014,ST015 | RS010,RS011 | GS010,GS011 | Northern loop from Fès through Rif mountains to Chefchaouen and back | Watch for rainfall on Rif descents; roads are grippy when dry, treacherous when wet | Cell coverage drops in Rif interior; inform contacts of route | Underrated northern circuit: green, cool, culture-rich |

---

## Route Descriptions

### RT001 — Grand Atlas Circuit
The Grand Atlas Circuit is the defining Morocco motorcycle experience — a complete loop from Marrakech that takes riders over the High Atlas, deep into the pre-Saharan desert, through the gorge corridors, and back. This is the route most adventure riders dream of before their first Morocco trip, and the route they recommend to everyone afterward.

**Character:** Mixed tarmac touring with a short adventure section approaching Merzouga. The riding balance favors technical mountain passes, long flowing desert plateau roads, and the intimate gorge corridors. Daily mileage runs 100–200km depending on detour choices and overnight selection. The route never becomes monotonous — the landscape transforms dramatically every 50–80km.

**Skill level:** Genuinely accessible to intermediate riders on road-capable bikes. The Merzouga approach on the N13 is paved; the sandy piste option is avoidable. Tizi n'Tichka requires confidence on tight switchbacks but not off-road skill. Beginners with some mountain pass experience can complete this route with care.

**Ideal bike:** Mid-size to large adventure bikes (BMW F750/850/1250GS, KTM 790/890 Adventure, Triumph Tiger, Royal Enfield Himalayan for lighter adventure approach) handle this route perfectly. Naked bikes and sports tourers are fine on the paved sections. Pure sportsbikes are manageable but the vibration on the plateau roads and limited luggage capacity reduce comfort.

**Isolation level:** Moderate. The main roads of the circuit (N9, N10, N13) see regular traffic — fuel and food are spaced 60–100km apart at worst. The Merzouga sector and gorge interiors are quieter and require more self-sufficiency. You are never more than 120km from a town with services on the main circuit.

**Fuel planning:** Critical section is the Erfoud–Merzouga gap (53km of good road) combined with limited fuel availability in Merzouga itself. Fill at Erfoud without exception before continuing east.

**Suggested overnight logic (7-day version):** Day 1 Marrakech (start/prepare). Day 2 cross Tichka to Ouarzazate. Day 3 Road of Kasbahs to Boumalne Dadès. Day 4 Dades and Todra gorges to Tinerhir. Day 5 to Merzouga via Erfoud. Day 6 Merzouga to Midelt via N13 and Ziz Gorge. Day 7 Midelt to Marrakech via Azrou or Khénifra.

**Seasonal notes:** October–November offers the best combination of manageable temperatures, low rainfall, and spectacular desert light. February–March is cool but clear and popular for dune photography. July–August requires extreme heat management — early starts, long midday rests, and serious hydration discipline.

---

### RT002 — Road of a Thousand Kasbahs
The N10 between Ouarzazate and Tinerhir is widely considered the most beautiful sustained stretch of paved road in Morocco. Running at 1000–1500m elevation through the pre-Saharan plateau, it threads a continuous sequence of ksar villages, palmeries, kasbah towers, rose gardens, and gorge entrances against a backdrop of ochre cliffs and Atlas ridgelines.

**Character:** Relaxed, scenic, culturally rich. This is a route for riders who want to ride slowly and stop often. The road quality is excellent — smooth tarmac, good sight lines, moderate traffic. Daily distances of 100–160km allow time to explore the gorges, visit kasbah interiors, and sit over long lunches. The vibe is supremely "motorcycle travel" rather than "motorcycle sport."

**Skill level:** Accessible to near-beginners on any touring or adventure bike. Road bikes are perfectly at home here. The only technical considerations are occasional sharp turns entering gorge sections and loose gravel on gorge interior roads.

**Scenic highlights:** The route enters Skoura's palmery immediately east of Ouarzazate (green, shaded, fragrant in spring). The approach to Boumalne Dadès reveals the Dadès Valley in full sweep. The Dadès Gorge hairpins (20km north of Boumalne) are the photographic highlight. The transition from plateau to Todra Valley palmery approaching Tinerhir is sublime. Each kasbah village along the N10 tells a different story of earth architecture.

**Food and overnight logic:** Auberges and guesthouses cluster at each gorge entrance — Skoura, Boumalne/Dadès Gorge, and Tinerhir/Todra Gorge all offer excellent overnight options. Lunch in the gorges themselves (small cafés at cliff base) is the experiential highlight. Never rush this route.

---

### RT003 — Tizi n'Test Adventure
The Tizi n'Test crossing is the most demanding and most rewarding paved mountain route in Morocco. The R203 from Marrakech to Taroudant climbs from the Haouz Plain through the Ourika Valley approaches, ascends the High Atlas in a series of increasingly dramatic and increasingly isolated switchbacks, crests at the Tizi n'Test pass (2092m), and descends through the Anti-Atlas foothills to the Souss Valley floor — a total south side drop of nearly 1800 meters.

**Character:** Serious mountain riding on a narrow, patched, sometimes crumbling paved road with minimal traffic, minimal infrastructure, minimal phone signal, and maximum scenery. This is a route for riders who accept risk as part of the experience. The road is legal and maintained, but it is fundamentally wild.

**Skill level:** Advanced riders only. The combination of narrow road, significant exposure (no guardrails on critical sections), variable surface quality, and zero rescue infrastructure demands genuine mountain riding competence. If you've never ridden a serious mountain pass before, do Tichka first.

**The Tin Mal reward:** The 12th-century Almohad mosque of Tin Mal, perched in its mountain valley 8km below the southern summit, is one of Morocco's most extraordinary historical sites and virtually unknown to the mass tourism circuit. Reaching it on a motorcycle on this road is one of the defining moments of Moroccan moto travel.

**Fuel planning:** Fill to the brim in Marrakech (or Asni, 47km south of Marrakech on the R203). The next guaranteed fuel is Ouled Berhil, 25km past Taroudant — a total dry stretch of approximately 200km. This is the longest guaranteed fuel gap on any major Moroccan road. Carry 2–3L reserve if your tank is under 18L.

---

### RT004 — Sahara Desert Express
The N13 from Fès to Merzouga is the great south-running highway of Morocco — long, fast, mostly straight, and dramatically shifting in landscape from green northern plateaus through arid anti-Atlas valleys to the absolute desert. For first-time Morocco riders, this is the ideal introduction: all paved, logical waypoints, manageable distances, and a spectacular payoff at the Erg Chebbi dunes.

**Character:** Long-distance touring with minimal technical challenge and maximum landscape variety. The road is wide and well-maintained for most of its length; the most dramatic section is the Gorges du Ziz near Rich, where the N13 threads through a deep canyon and past the Hassan Addakhil dam. From Midelt south, the landscape becomes increasingly spare and beautiful.

**Skill level:** Accessible to beginners with basic highway confidence. The route can be done in two long days (Fès–Midelt, Midelt–Merzouga) or three comfortable days. Traffic is light south of Midelt.

**The Ziz highlight:** The Gorges du Ziz section (between Midelt and Errachidia) should not be rushed. The tunnels through the gorge were carved during the French Protectorate — use your headlight and horn at each tunnel entrance. The dam lake above Errachidia is photogenic and offers a good rest stop.

**Overnight logic:** Midelt for night 1 (all services, cool plateau atmosphere). Erfoud area or Rissani for night 2 (closer to Merzouga). Merzouga itself for the desert sunrise/sunset experience.

---

### RT005 — Atlantic Coast & Anti-Atlas
This route combines coastal Morocco, the agricultural Souss Valley, and the dramatic Tizi n'Test mountain crossing — three completely different Moroccan landscapes in one two-to-three-day circuit. Riding south from Agadir along the Atlantic coast (N1 south, then P1814 inland) to Taroudant, then north over Tizi n'Test back to Marrakech, creates a logical and deeply satisfying triangular route.

**Character:** Varies dramatically by section. The coastal leg is flat and fast (good road, Atlantic breeze, argan tree landscapes). The Souss Valley crossing is gentle and warm. The Tizi n'Test climb is intense and isolated. The variety prevents the route from feeling long.

**Skill level:** The coastal and valley sections are easy. The Test pass demands advanced skill. This route should only be attempted by riders who are confident on technical mountain roads. Consider riding the Test in this direction (north to south from Taroudant to Marrakech) as it involves more of the scary section on the valley floor side — arguably more controllable.

---

### RT006 — Rif Mountains Loop
The Rif Mountains loop is Morocco's most underrated motorcycle circuit. Running from Fès north through Taza and Ketama to Chefchaouen and back via Ouazzane, it explores a completely different Morocco from the Atlas and desert circuits — green, forested, sometimes lush, historically complex, and cooler than almost anywhere else in the country.

**Character:** Winding mountain roads through cedar and oak forest, Mediterranean-influenced villages, dramatic Rif plateau views, and the visual payoff of Chefchaouen's blue medina. Road quality is mixed — good on main N axes, variable on secondary mountain routes. Rain is possible April–May and October–November; roads become significantly slippery.

**Skill level:** Intermediate. The Rif roads demand attention but not extreme technical skill. The main hazard is wet road surfaces on mountain descents — tire choice matters here. Semi-knobby or touring tires outperform pure sport compounds in these conditions.

**Chefchaouen overnight:** Non-negotiable. The town rewards slowing down — a walk through the medina at dusk, dinner on a terrace, and the mountain panorama at dawn are among Morocco's finest experiences for travelers of any kind.

---

# SECTION 3: ROUTE SEGMENTS

## Route Segments Table

| segment_id | route_id | segment_order | segment_name | start_latitude | start_longitude | end_latitude | end_longitude | distance_km | estimated_duration | surface_type | difficulty_level | hazard_notes | scenic_notes | nearest_place_id | nearest_restaurant_id | nearest_stay_id | nearest_fuel_station_id | stop_recommendation |
|------------|----------|---------------|-------------|----------------|-----------------|--------------|---------------|-------------|-------------------|--------------|-----------------|--------------|--------------|-----------------|----------------------|-----------------|------------------------|---------------------|
| SEG001 | RT001 | 1 | Marrakech to Aït Ourir | 31.628 | -7.992 | 31.565 | -7.620 | 36 | 45 min | paved, good | easy | busy N9 exit from Marrakech, pedestrians | olive groves, Atlas foothills appear | PL001 | RS001 | ST001 | GS001 | fuel up before departure |
| SEG002 | RT001 | 2 | Aït Ourir to Tizi n'Tichka Summit | 31.565 | -7.620 | 31.253 | -7.374 | 62 | 90 min | paved, winding | intermediate | tight switchbacks, blind bends, occasional loose gravel | dramatic High Atlas ascent, Berber villages | PL007 | RS001 | ST001 | GS001 | pause at summit for photos and tea |
| SEG003 | RT001 | 3 | Tichka Summit to Ouarzazate | 31.253 | -7.374 | 30.920 | -6.893 | 62 | 80 min | paved, good | easy-intermediate | long descent, watch for trucks | vast plateau view, anti-atlas backdrop | PL002 | RS002 | ST002 | GS002 | refuel and rest in Ouarzazate |
| SEG004 | RT001 | 4 | Ouarzazate to Skoura Oasis | 30.920 | -6.893 | 31.063 | -6.559 | 42 | 45 min | paved, excellent | easy | minimal | palmery entry, first kasbah views | PL017 | RS002 | ST003 | GS002 | optional auberge detour into palmery |
| SEG005 | RT001 | 5 | Skoura to Boumalne Dadès | 31.063 | -6.559 | 31.371 | -5.978 | 55 | 60 min | paved, excellent | easy | minimal | rose valleys, kasbah towers, valley floor | PL018 | RS004 | ST007 | GS005 | continue or overnight here |
| SEG006 | RT001 | 6 | Boumalne to Dades Gorge Hairpins | 31.371 | -5.978 | 31.523 | -6.017 | 25 | 40 min | paved, good (narrows upper) | intermediate | hairpin corners, possible gravel | world-famous monkey-finger hairpins | PL006 | RS005 | ST008 | GS005 | stop at upper viewpoint, photograph |
| SEG007 | RT001 | 7 | Boumalne to Tinerhir (N10 east) | 31.371 | -5.978 | 31.515 | -5.524 | 60 | 65 min | paved, excellent | easy | minimal | broad valley, kasbah clusters | PL011 | RS004 | ST009 | GS005 | refuel in Tinerhir |
| SEG008 | RT001 | 8 | Tinerhir to Todra Gorge | 31.515 | -5.524 | 31.591 | -5.600 | 14 | 20 min | paved, good | easy | narrow in gorge, tourist buses | palmery approach then canyon drama | PL005 | RS003 | ST006 | GS005 | lunch stop in gorge |
| SEG009 | RT001 | 9 | Tinerhir to Erfoud (N10) | 31.515 | -5.524 | 31.434 | -4.235 | 100 | 110 min | paved, very good | easy | long straight, fatigue risk | flat Tafilalet plains, palmeries | PL010 | RS008 | ST011 | GS003 | refuel in Erfoud — critical |
| SEG010 | RT001 | 10 | Erfoud to Merzouga | 31.434 | -4.235 | 31.101 | -4.013 | 53 | 55 min | paved N13 | easy | last km near dunes: soft sand possible | dunes appear on horizon 15km out | PL004 | RS009 | ST005 | GS003 | overnight in Merzouga |
| SEG011 | RT001 | 11 | Merzouga to Rissani | 31.101 | -4.013 | 31.280 | -4.260 | 40 | 45 min | paved, fair | easy | sandy sections at start | desert exit, date palms | PL019 | RS008 | ST012 | GS008 | top up fuel if needed |
| SEG012 | RT001 | 12 | Rissani to Midelt via Ziz Gorge | 31.280 | -4.260 | 32.685 | -4.733 | 175 | 2h 40min | paved excellent, gorge tunnels | easy-intermediate | gorge tunnels (use lights), long straight plateau | Ziz Gorge drama, dam lake, plateau cool air | PL009 | RS003 | ST011 | GS008 | lunch in Midelt |
| SEG013 | RT001 | 13 | Midelt to Marrakech via Azrou | 32.685 | -4.733 | 31.628 | -7.992 | 280 | 3h 30min | paved, good-excellent | easy-intermediate | cedar forest (Azrou), busy Casablanca approach | Middle Atlas cedar forest, Khenifra valley | PL001 | RS001 | ST001 | GS001 | final leg, strong finish |
| SEG014 | RT003 | 1 | Marrakech to Asni | 31.628 | -7.992 | 31.248 | -7.979 | 47 | 55 min | paved, good | easy | N-R203 junction navigation | Haouz plain, Atlas approaches, Ourika Valley | PL001 | RS001 | ST001 | GS006 | last reliable fuel before Test |
| SEG015 | RT003 | 2 | Asni to Tizi n'Test Summit | 31.248 | -7.979 | 30.870 | -8.503 | 80 | 2h | paved, narrow, patched | advanced | no guardrails, blind bends, rockfall possible | extreme Atlas drama, cliff-cut road | PL014 | RS006 | ST009 | GS006 | slow and focused; stop at Tin Mal |
| SEG016 | RT003 | 3 | Tizi n'Test Summit to Taroudant | 30.870 | -8.503 | 30.472 | -8.876 | 75 | 90 min | paved, improving on descent | intermediate | first 20km still narrow | Sous Valley panorama from above | PL015 | RS007 | ST010 | GS007 | overnight Taroudant |
| SEG017 | RT004 | 1 | Fès to Midelt (N13 south) | 34.037 | -5.000 | 32.685 | -4.733 | 220 | 2h 45min | paved, excellent | easy | minimal | Middle Atlas, cedar forests, Ifrane | PL009 | RS003 | ST011 | GS010 | midpoint lunch or overnight |
| SEG018 | RT004 | 2 | Midelt through Ziz Gorge to Errachidia | 32.685 | -4.733 | 31.929 | -4.424 | 110 | 1h 30min | paved, excellent, gorge tunnels | easy-intermediate | gorge tunnels use lights | Ziz Gorge, dam lake | PL009 | RS008 | ST012 | GS008 | don't rush gorge section |
| SEG019 | RT004 | 3 | Errachidia to Erfoud | 31.929 | -4.424 | 31.434 | -4.235 | 55 | 50 min | paved, good | easy | minimal, crosswinds on plateau | Tafilalet approaches | PL010 | RS008 | ST011 | GS003 | refuel Erfoud |
| SEG020 | RT004 | 4 | Erfoud to Merzouga | 31.434 | -4.235 | 31.101 | -4.013 | 53 | 55 min | paved N13 | easy | dune access tracks sandy | dunes visible from 15km | PL004 | RS009 | ST005 | GS003 | overnight |

---

# SECTION 4: HOTELS / AUBERGES / STAYS

## Stays Table

| stay_id | name | place_id | region | latitude | longitude | stay_type | price_range | facilities | motorbike_friendly | secure_parking | hot_shower | breakfast_available | private_room_available | group_friendly | booking_recommended | description | ideal_for |
|---------|------|----------|--------|----------|-----------|-----------|-------------|------------|-------------------|---------------|------------|--------------------|-----------------------|----------------|--------------------|---------|----|
| ST001 | Riad Porte Royale | PL001 | Marrakech-Safi | 31.624 | -7.987 | riad | €50-80 | WiFi, AC, breakfast, pool | yes | covered garage (fee) | yes | yes | yes | yes | yes | Elegant riad near Bab Doukkala; garage for bikes available 200m away | touring groups, start/end of trip |
| ST002 | Hotel Azoul Ouarzazate | PL002 | Draa-Tafilalet | 30.918 | -6.895 | hotel | €35-55 | WiFi, AC, breakfast, pool | yes | courtyard parking | yes | yes | yes | yes | sometimes | Reliable mid-range hotel on main boulevard with motorcycle-accessible courtyard | all riders, southern circuit |
| ST003 | Auberge Dar Ahlam Palmeraie | PL017 | Draa-Tafilalet | 31.062 | -6.555 | auberge | €25-45 | breakfast, dinner available | yes | open courtyard | yes | yes | yes | yes | recommended | Atmospheric palmery auberge inside Skoura oasis; sandy floor parking | romance riders, culture seekers |
| ST004 | Camping-Auberge Chez Julia | PL004 | Draa-Tafilalet | 31.099 | -4.009 | auberge | €20-40 | dinner, WiFi, terrace, dune view | yes | sand yard (flat stand needed) | basic | yes | yes | yes | yes (Oct-Mar) | Friendly dune-edge auberge with legendary sunset terrace | desert seekers, solo riders |
| ST005 | Kasbah Mohayut | PL004 | Draa-Tafilalet | 31.103 | -4.016 | kasbah_hotel | €60-100 | pool, AC, restaurant, WiFi | yes | guarded yard | yes | yes | yes | yes | highly recommended | Best mid-range Merzouga option; reliable, consistent, secure | groups, couples, budget-plus |
| ST006 | Auberge Les Roches Todra | PL005 | Draa-Tafilalet | 31.589 | -5.598 | auberge | €20-35 | dinner, terrace, gorge view | yes | roadside parking | cold only | yes | yes | yes | recommended | Cliff-base auberge inside the gorge; extraordinary location, basic comfort | adventure riders, solo travelers |
| ST007 | Hotel Salam Boumalne | PL018 | Draa-Tafilalet | 31.370 | -5.975 | hotel | €25-40 | breakfast, WiFi, terrace | yes | street parking (reliable) | yes | yes | yes | yes | not usually | Good base for gorge exploration; clean, functional, honest | all riders |
| ST008 | Auberge Tissadrin Dades | PL006 | Draa-Tafilalet | 31.521 | -6.014 | auberge | €25-40 | dinner, gorge view, WiFi | yes | roadside gravel yard | yes | yes | yes | yes | recommended Apr-Oct | Set in middle gorge section with direct hairpin views; most atmospheric Dadès overnight | all riders, scenic priority |
| ST009 | Auberge Aoullouz | PL014 | Marrakech-Safi | 30.872 | -8.505 | auberge | €15-25 | basic room, dinner | yes | roadside | no | no | yes | no | not needed | Simple mountain stop near Tizi n'Test summit for early starters | adventure riders |
| ST010 | Riad Taroudant | PL015 | Souss-Massa | 30.470 | -8.875 | riad | €45-70 | WiFi, breakfast, courtyard garden | yes | nearby guarded parking | yes | yes | yes | yes | recommended | Lovely mid-range riad in Taroudant medina; relaxed garden atmosphere | all riders, culture travelers |
| ST011 | Hotel Ayachi Midelt | PL009 | Draa-Tafilalet | 32.683 | -4.732 | hotel | €20-35 | WiFi, breakfast, terrace | yes | guarded rear lot | yes | yes | yes | yes | not usually | Best value hotel in Midelt; motorcycle groups often use this | trans-Atlas riders, groups |
| ST012 | Kasbah Hotel Erfoud | PL010 | Draa-Tafilalet | 31.432 | -4.233 | hotel | €25-40 | pool, AC, WiFi, breakfast | yes | guarded lot | yes | yes | yes | yes | sometimes | Convenient pre-Merzouga overnight; good pool in summer if you must transit | all riders |
| ST013 | Hotel Agadir Beach Club | PL016 | Souss-Massa | 30.425 | -9.595 | hotel | €65-110 | pool, AC, secure parking, WiFi | yes | underground parking | yes | yes | yes | yes | yes | Good quality coastal stop for tour circuits using Agadir as logistics hub | groups, service-day stays |
| ST014 | Dar Seffarine Fès | PL012 | Fès-Meknès | 34.064 | -4.974 | riad | €60-90 | WiFi, breakfast, AC | yes | parking nearby (PL) | yes | yes | yes | yes | yes | Comfortable medina-edge riad; helpful owners for bike parking advice | culture travelers, northern circuits |
| ST015 | Hotel Parador Chefchaouen | PL013 | Tanger-Tetouan | 35.169 | -5.270 | hotel | €50-80 | WiFi, breakfast, pool | yes | outdoor guarded lot | yes | yes | yes | yes | recommended Apr-Jun | Classic Chefchaouen hotel; reliable, mid-range, good views | all riders, Rif circuit |

---

## Stay Descriptions

### ST001 — Riad Porte Royale, Marrakech
A polished but unpretentious riad near Bab Doukkala, the western gate of the medina. Positioned on the medina edge where the old city transitions to quieter residential streets — navigable on a large bike without entering the souk chaos. The owners can arrange motorcycle parking in a covered garage 200m away (€5–8/night) through a reliable contact. WiFi is fast and the breakfast (served on a rooftop terrace with Atlas views on clear days) is the kind of meal that sets a ride up well. For groups of 3–6 riders, the riad can be rented entirely. Book at least 2 weeks ahead for October–November and February–March.

### ST002 — Hotel Azoul Ouarzazate
The default mid-range hotel for motorcycle travelers in Ouarzazate — everyone who has done the south Morocco circuit seems to have stayed here at least once. The courtyard can accept 6–8 bikes flat on tarmac. Staff understand motorcycle travel and can help with local repair contacts. The rooftop restaurant serves decent tagines; the bar (unusually for Morocco) is reliable. Location on Avenue Mohammed V puts fuel, ATMs, and the main kasbah walk within easy riding distance.

### ST003 — Auberge Dar Ahlam Palmeraie, Skoura
Hidden inside the Skoura palmery on a sandy track accessible from the main N10 (signposted). The last 500m requires slow, careful navigation on sand — heavy loaded bikes benefit from having a second rider walk ahead to check the path. Once there, the setting is magical: date palms overhead, a rose garden, and complete quiet. Dinner (€12–15 for full Berber menu) is the highlight. Sandy courtyard is flat but soft — use your center stand's side pad or a block under the stand foot.

### ST004 — Camping-Auberge Chez Julia, Merzouga
The most character-filled budget option at the Erg Chebbi dune edge. Julia (French-Moroccan management) runs a guesthouse of legendary reputation among solo moto travelers and backpackers. The terrace facing the dunes is the place to be at sunset — cold drinks, Saharan silence, 150m of golden sand wall turning orange. Dorm beds (€10) and simple private rooms (€20–30). The motorcycle yard is sandy — a side stand plate is essential for heavy bikes. Book absolutely in advance October–November and February.

### ST005 — Kasbah Mohayut, Merzouga
Step up in comfort from the basic auberges: proper air conditioning, a tiled pool, WiFi that actually works, and an on-site restaurant with an Erg Chebbi view. The owners are experienced with motorcycle groups — the guarded parking area has been hardened with compacted gravel and can accept 10+ bikes. Reliable hot showers (crucial after desert grime). Price includes breakfast. Good dinner option at the restaurant (€15–20). Best option for riders who want desert atmosphere without sacrificing basic comfort.

### ST006 — Auberge Les Roches Todra
Literally built against the cliff wall inside the Todra Gorge. The walls of the canyon are your view from every room. Cold water only (hot shower option requires booking in advance and a small supplement). The roadside parking area is narrow but usable — 4–5 bikes max. This is not a comfortable hotel; it is an experience. Lying in bed listening to the canyon wind while damp gear dries on the terrace is a motorcycle travel memory that lasts.

### ST007 — Hotel Salam Boumalne
Clean, functional, honest. No luxury claims, no Instagram-worthy decor — just reliable beds, working showers, a pleasant terrace with Dadès Valley views, and an owner who will keep an eye on your bikes without being asked. The town center location means fuel, a basic supermarket, and several restaurants are within walking distance. Good value for a practical gorge-exploration overnight.

### ST011 — Hotel Ayachi Midelt
The reliable midpoint hotel for the classic Fès–Merzouga run. Motorcycle tour groups from Europe regularly use this as their Midelt overnight — the guarded rear parking lot has been purpose-expanded and can accept 15+ bikes. Staff speak French and some English. The breakfast spread is substantial. The hotel restaurant is mediocre — walk to the town's market street instead for better tagine at half the price.

---

# SECTION 5: RESTAURANTS / FOOD STOPS

## Restaurants Table

| restaurant_id | name | place_id | region | latitude | longitude | cuisine_type | price_level | vibe | biker_friendly | quick_stop_friendly | group_friendly | meal_type | description | ideal_for |
|---------------|------|----------|--------|----------|-----------|--------------|-------------|------|----------------|--------------------|--------------|-----------|-----------|----|
| RS001 | Café-Restaurant Atlas Aït Ourir | PL001 | Marrakech-Safi | 31.564 | -7.620 | Moroccan | low | roadside, functional | yes | yes | yes | breakfast/lunch | Roadside café at the Aït Ourir junction; harira, msemen, coffee — the classic pre-climb breakfast | pre-Tichka fuel stop |
| RS002 | Restaurant Chez Mimoun | PL002 | Draa-Tafilalet | 30.918 | -6.890 | Moroccan | low-medium | local, authentic | yes | no | yes | lunch/dinner | Best tagine in Ouarzazate according to local motorcycle guides; slow-cooked lamb, no tourist menu | Ouarzazate overnight |
| RS003 | Café Tinguit | PL009 | Draa-Tafilalet | 32.681 | -4.731 | Moroccan | low | market café | yes | yes | yes | lunch | Classic market-street café in Midelt; brochettes, harira, fresh bread | midpoint refuel |
| RS004 | Auberge Restaurant Vallée des Roses | PL018 | Draa-Tafilalet | 31.368 | -5.977 | Moroccan | low | scenic terrace | yes | no | yes | lunch/dinner | Terrace restaurant with Dadès Valley panorama; tagines and salads at honest prices | lunch between gorges |
| RS005 | Kasbah Café des Roses | PL017 | Draa-Tafilalet | 31.061 | -6.553 | Moroccan | low | palmery garden | yes | no | no | lunch | Shaded garden café inside Skoura palmery; juice, mint tea, simple Berber dishes | slow lunch detour |
| RS006 | Café Tin Mal | PL014 | Marrakech-Safi | 30.871 | -8.504 | Moroccan | very low | mountain simple | yes | yes | no | lunch | Tiny café adjacent to Tin Mal mosque; mint tea, basic sandwiches — atmosphere is everything | mid-pass stop |
| RS007 | Restaurant la Gazelle Taroudant | PL015 | Souss-Massa | 30.470 | -8.876 | Moroccan | low-medium | souk-edge terrace | yes | no | yes | lunch/dinner | Excellent lamb with prunes and preserved lemon; souk-edge location with rampart views | Taroudant overnight |
| RS008 | Restaurant Le Palmier Erfoud | PL010 | Draa-Tafilalet | 31.431 | -4.234 | Moroccan | low | local, simple | yes | yes | yes | lunch | Reliable town-center restaurant; serves fast tagines good for riders with time pressure | quick Erfoud stop |
| RS009 | Café-Restaurant Erg Chebbi | PL004 | Draa-Tafilalet | 31.100 | -4.012 | Moroccan/International | low-medium | dune-view terrace | yes | no | yes | dinner | Dune-view terrace restaurant; not sophisticated but the setting compensates entirely | Merzouga sunset dinner |
| RS010 | Restaurant Tissemane Fès | PL012 | Fès-Meknès | 34.063 | -4.975 | Moroccan | medium | medina courtyard | yes | no | yes | dinner | Mid-range medina restaurant with proper Fassi cuisine; bastila, mrouzia, lamb dishes | Fès cultural dinner |
| RS011 | Café Clock Chefchaouen | PL013 | Tanger-Tetouan | 35.167 | -5.267 | international/Moroccan | low-medium | hip medina | yes | no | yes | lunch/dinner | Relaxed café with good coffee, salads, and mountain views; great for rest-day lunches | Chefchaouen stay |

---

## Restaurant Descriptions

### RS001 — Café-Restaurant Atlas, Aït Ourir
The essential pre-climb breakfast stop before Tizi n'Tichka. Located at the Aït Ourir junction on the N9, this roadside café (and every similar one in the town's main street) serves the Moroccan morning staple: harira soup, msemen flatbreads with amlou (almond-argan paste), mint tea, and espresso. Riders who skip breakfast in Marrakech and stop here instead avoid city traffic stress and start the Atlas climb well-fueled. Most of the town's cafés are indistinguishable — the first one you spot with tables outside and motorbikes parked in front will serve the purpose.

### RS002 — Restaurant Chez Mimoun, Ouarzazate
An underground recommendation passed between motorcycle travelers for years. Hidden on a side street off the main boulevard (ask locals, it moves occasionally), Chez Mimoun serves slow-cooked tagine from a clay pot — lamb, prunes, and almonds or chicken with preserved lemon and olives — that is substantially better than the tourist restaurant menus. No English menu, basic French spoken. A meal here costs €6–9. Book nothing; just arrive and wait for a table if needed.

### RS004 — Auberge Restaurant Vallée des Roses, Boumalne Dadès
A terrace restaurant elevated slightly above the N10 with a sweeping view down the Dadès Valley. The kitchen runs on tagine and couscous Friday) — nothing extraordinary, but honest cooking and reliable quality at €7–10 per meal. The terrace is an ideal lunch stop between Ouarzazate and Tinerhir: pull off, park the bike in the shade, eat with a view, recover for the afternoon's gorge riding. Usually open 11am–4pm; call ahead for dinner (some close evenings outside peak season).

### RS009 — Café-Restaurant Erg Chebbi, Merzouga
The best thing about this restaurant is not the food (adequate tagines, some international dishes) but where you eat it: a rooftop terrace 20m from the dune edge, watching the Erg Chebbi turn gold, orange, and purple as the sun drops. Order the mechoui (slow-roasted lamb, available by arrangement) for groups. Otherwise, standard tagines €8–12. Bring a light layer — the dune zone cools dramatically at sunset.

---

# SECTION 6: GAS STATIONS

## Gas Stations Table

| station_id | name | place_id | region | latitude | longitude | fuel_types | open_24h | payment_notes | reliability_notes | air_pump_available | basic_repair_nearby | description |
|------------|------|----------|--------|----------|-----------|------------|----------|---------------|-------------------|--------------------|---------------------|-|
| GS001 | Afriquia Marrakech Nord | PL001 | Marrakech-Safi | 31.645 | -7.985 | SP95, SP98, diesel | yes | cash and card | very reliable, chain station | yes | yes (moto shop nearby) | Last major station before N9 south; stock up before Atlas |
| GS002 | Ziz Energie Ouarzazate | PL002 | Draa-Tafilalet | 30.922 | -6.888 | SP95, diesel | no (7am-9pm) | cash preferred | reliable, multiple pumps | yes | yes | Main boulevard station; always queue in morning with tour groups |
| GS003 | Total Erfoud | PL010 | Draa-Tafilalet | 31.435 | -4.232 | SP95, SP98, diesel | no (7am-8pm) | cash | reliable | yes | no | Critical desert refuel point before Merzouga — do not skip |
| GS004 | Afriquia Tinerhir | PL011 | Draa-Tafilalet | 31.514 | -5.522 | SP95, diesel | no (7am-9pm) | cash | reliable | yes | small repair shop nearby | Standard stop on N10 corridor; fill up before Dades/Todra gorge sectors |
| GS005 | Station Nationale Boumalne | PL018 | Draa-Tafilalet | 31.373 | -5.976 | SP95, diesel | no (7am-8pm) | cash | reliable | no | basic | Good N10 fill-up between Ouarzazate and Tinerhir |
| GS006 | Afriquia Asni | PL014 | Marrakech-Safi | 31.249 | -7.980 | SP95, diesel | no (7am-8pm) | cash | reliable but sometimes low stock | no | no | Last fuel before Tizi n'Test; critical fill point — verify stock |
| GS007 | Station Ouled Berhil | PL015 | Souss-Massa | 30.455 | -8.802 | SP95, diesel | no (7am-7pm) | cash | moderate reliability | no | no | First station south of Tizi n'Test; may have queues or low stock |
| GS008 | Ziz Station Rissani | PL019 | Draa-Tafilalet | 31.282 | -4.258 | SP95, diesel | no (8am-7pm) | cash only | moderate, varies by day | no | basic repair | Market-day fuel availability better; backup for Merzouga-bound riders |
| GS009 | Agadir Total Central | PL016 | Souss-Massa | 30.427 | -9.596 | SP95, SP98, diesel | yes | cash and card | very reliable | yes | yes (bike shop 500m) | Major city station; all fuels, full service, near moto workshop |
| GS010 | Afriquia Fès Sud | PL012 | Fès-Meknès | 34.032 | -4.997 | SP95, SP98, diesel | yes | cash and card | very reliable | yes | yes | South Fès ring road station; good for departure fill before N13 |
| GS011 | Marjane Station Fès | PL012 | Fès-Meknès | 34.028 | -4.998 | SP95, diesel | no (8am-10pm) | card accepted | reliable | yes | no | Hypermarket-adjacent; useful for supplies alongside fuel |

---

## Gas Station Descriptions

### GS001 — Afriquia Marrakech Nord
A full-service chain station on the N9 north of Marrakech's main ring road — the most practical fill point before heading south toward Tizi n'Tichka or southwest toward Agadir. Open 24 hours, card payments accepted, air pump operational. The adjacent motorcycle supplies shop (100m north) sells basic spares — inner tubes, chain lube, bungee cords, bulbs — useful for last-minute gear checks before a long desert loop. Very busy in the morning with tour groups and local traffic; arrive before 7:30am or expect queues.

### GS002 — Ziz Energie Ouarzazate
The primary fuel stop in Ouarzazate for motorcyclists, located on the main commercial boulevard. Multiple pumps handle the significant volume of transit traffic (tourist buses, 4x4 convoys, local taxis). SP95 and diesel are consistently available. Attendants are experienced with motorcycle tanks — they know not to overflow and generally understand the premium fuel preference. A well-stocked convenience store adjacent is useful for water, snacks, and emergency supplies. Cash preferred; card readers have been reported as occasionally non-functional.

### GS003 — Total Erfoud
The most critical fuel stop in the entire southern Morocco desert circuit. Erfoud is the last guaranteed fuel source before Merzouga (53km) and before the more isolated piste routes east and south. The Total station on the N13 main axis is reliable and generally well-stocked, but it operates standard hours (not 24H) — arriving after 8pm may find it closed. Fill absolutely full here without exception. If a queue exists, wait. This is not a fill-point to skip. Cash recommended; the card terminal occasionally fails.

### GS006 — Afriquia Asni
The last fuel stop before the Tizi n'Test crossing, located in Asni village on the R203. Stock levels here are variable — this is a small-capacity rural station, and in peak season (April–May, October) it can run low on SP95 by afternoon. Fill in Marrakech as the primary strategy and treat Asni as a top-up opportunity. If the station is closed or dry, the next source is Ouled Berhil on the far south side of the pass (200+ km away). Do not gamble with this.

---

# SECTION 7: WEATHER RISKS

## Weather Risks Table

| risk_id | related_route_id | related_place_id | region | season | risk_type | severity | description | rider_advice |
|---------|-----------------|-----------------|--------|--------|-----------|----------|-------------|--------------|
| WR001 | RT001 | PL007 | Marrakech-Safi | Nov-Mar | snow_and_ice | high | Tizi n'Tichka (2260m) receives significant snowfall Nov-Mar; pass can close without warning | Check conditions daily; authorities close gate at pass base when snow is forecast; alternative via N9 bypass is long |
| WR002 | RT003 | PL014 | Marrakech-Safi | Nov-Mar | snow_and_ice | very high | Tizi n'Test is narrower and less maintained than Tichka; ice on shaded sections persists for days after snowfall | Do not attempt Oct-Apr without confirmed clear conditions; no rescue infrastructure |
| WR003 | RT001 | PL004 | Draa-Tafilalet | Jun-Aug | extreme_heat | very high | Merzouga and Erg Chebbi area regularly exceeds 48°C Jun-Aug; road surface deforms; hydration risk extreme | Ride only before 9am and after 5pm; carry 3L water per rider; avoid dark gear; check engine temp |
| WR004 | RT001 | PL003 | Draa-Tafilalet | Jun-Aug | extreme_heat | high | Zagora regularly hits 45°C in summer; heat exhaustion risk increases with full riding gear | Mandatory midday stop in shade; plan all fuel and food stops for early morning |
| WR005 | RT001 | PL005 | Draa-Tafilalet | Mar-May, Sep | flash_flood | high | Todra Gorge is subject to flash floods when rainfall occurs on distant Atlas ridges; zero local warning | Vacate gorge interior immediately if sky darkens upstream or if water level rises; never camp in gorge bed |
| WR006 | RT001 | PL006 | Draa-Tafilalet | Mar-May | flash_flood | medium | Dades Gorge has same flash flood risk as Todra; upper section is narrower and drainage is faster | Same precautions as Todra; avoid gorge interior in afternoon if thunderstorm risk exists on Atlas |
| WR007 | RT002 | PL010 | Draa-Tafilalet | Mar-May | sandstorm | medium | Spring khamsin winds drive sand from the Sahara northward; visibility can drop to 50-100m | Stop and wait out the storm; pull off road completely; cover air filter opening with cloth |
| WR008 | RT004 | PL009 | Draa-Tafilalet | Dec-Feb | mountain_cold | medium | Midelt at 1508m can see temperatures of -5 to +5°C Dec-Feb with occasional light snow | Carry full winter layers; glove liners; avoid dawn starts in sub-zero conditions |
| WR009 | RT001,RT003 | PL007,PL014 | Multi | Oct-Apr | fog | medium | Morning fog on High Atlas passes reduces visibility to 50-150m; descents are particularly dangerous | Delay departure until fog clears; use low beam and drive slowly; horn at every blind turn |
| WR010 | RT001 | PL007 | Marrakech-Safi | year-round | crosswind | medium | Exposed sections of Tizi n'Tichka summit area subject to strong crosswinds especially in spring and autumn | Maintain relaxed grip; watch for wind tunnels at ridge cuts; loaded bikes are more stable than empty ones |
| WR011 | RT003 | PL014 | Marrakech-Safi | May-Sep | rockfall | medium | Tizi n'Test road is subject to rockfall particularly after rain; loose stones accumulate in blind corners | Ride inside line on blind corners to allow visibility and reaction time; slow down after any rain |
| WR012 | RT006 | PL013 | Tanger-Tetouan | Oct-Mar | wet_roads | medium | Rif Mountains receive significant rainfall in winter; mountain roads become extremely slippery particularly on descents | Use intermediate pace on Rif descents regardless of dry appearance; tires cool rapidly in shadow zones |
| WR013 | RT001 | PL004 | Draa-Tafilalet | Feb-Apr | sandstorm | medium | Erg Chebbi area subject to spring sandstorms of 1-3 day duration; air filter contamination risk | Cover air intake when stopped; carry extra air filter; check oil level after storm passage |

---

# SECTION 8: RIDER MATCHING RULES

## Overview
The AI recommendation system uses a multi-factor matching process to align route recommendations with rider profiles. Each rider is assessed across several dimensions, and routes, stays, and food stops are selected or filtered based on the resulting compatibility scores.

## Profile Input Variables

| variable | type | values | impact |
|----------|------|--------|--------|
| rider_experience | categorical | beginner / intermediate / advanced / expert | primary route filter |
| bike_type | categorical | road/sport / touring / adventure / dual-sport / scooter | route and segment filter |
| group_size | integer | 1-15 | stay and parking filter |
| budget_per_day | numeric (EUR) | <30 / 30-60 / 60-100 / 100+ | stay and restaurant filter |
| trip_duration_days | integer | 1-21 | route length filter |
| preferred_vibe | categorical | culture / adventure / desert / coastal / scenic / mixed | route character filter |
| sleep_preference | categorical | budget/auberge / mid-range / comfort / camping | stay filter |
| country_of_origin | string | ISO country | language and cultural notes |
| daily_ride_km_tolerance | numeric | <100 / 100-150 / 150-200 / 200+ | segment pacing filter |
| physical_condition | categorical | standard / reduced_mobility / athletic | comfort and fatigue notes |

---

## Matching Rules by Category

### Rule Set A: Experience-Based Route Filtering

**Rule A1 — Beginner riders (experience = beginner):**
- Eligible routes: RT002 (Kasbahs Road), RT004 (Sahara Express)
- Ineligible routes: RT003 (Tizi n'Test), RT006 (Rif Mountains), any route with segments rated "advanced"
- Override condition: beginner rider with an experienced guide companion → unlock intermediate routes with mandatory daily distance cap of 150km
- Note: Merzouga dune-approach sand piste segments must be routed via paved N13 alternative only

**Rule A2 — Intermediate riders (experience = intermediate):**
- Eligible routes: RT001 (Grand Atlas), RT002 (Kasbahs Road), RT004 (Sahara Express), RT006 (Rif Loop) with weather check
- Ineligible routes: RT003 (Tizi n'Test) without explicit preference for advanced mountain riding
- Trigger RT003 only if: intermediate + adventure bike + preferred_vibe includes "mountain" + trip_duration ≥ 5 days

**Rule A3 — Advanced/Expert riders:**
- All routes eligible
- RT003 (Tizi n'Test) recommended if preferred_vibe = adventure or mountain
- High-pass routes flagged with seasonal weather check regardless of skill level

---

### Rule Set B: Bike-Type Compatibility

**Rule B1 — Road/sport bikes (touring, naked, sport):**
- Suitable for: RT001 main circuit (paved sections only), RT002, RT004, RT005 (coastal-valley only)
- Exclude: any segment with surface_type = piste or sand
- Merzouga approach: must use paved N13; sandy dune tracks excluded
- RT003: technically permitted but comfort warning issued (narrow patched road on aggressive geometry)

**Rule B2 — Adventure/dual-sport bikes:**
- Fully eligible for all routes including piste options
- RT001 with piste variant to Msemrir (PL020) unlocked for experienced adventure riders
- Note: heavy adventure bikes (≥230kg loaded) should avoid steep sandy piste sections; recommend 21" front wheel minimum for piste segments

**Rule B3 — Scooters (125cc+):**
- Eligible for RT002 and RT004 main corridor
- Exclude all mountain passes (Tichka, Test) — power and braking insufficient for steep grades
- Fuel planning adjusted: smaller tanks require more frequent stops; flag 90km+ fuel gaps as critical

---

### Rule Set C: Budget-Based Accommodation and Food Selection

**Rule C1 — Budget travelers (≤€30/day):**
- Stay selection: auberges with shared bathroom, camping options, dorm rooms
- Recommended stays: ST004 (Chez Julia), ST006 (Les Roches), ST009 (Aoullouz)
- Food selection: roadside cafés, market restaurants
- Adjust itinerary to reduce city overnight costs (avoid Marrakech Riad premium nights)

**Rule C2 — Mid-range travelers (€30-80/day):**
- Stay selection: standard hotels, mid-range riads, comfortable auberges
- Full stay database accessible
- Budget allocation: allow premium overnight at Merzouga (ST005) due to unique desert context

**Rule C3 — Comfort travelers (€80+/day):**
- Prefer riads, kasbah hotels, properties with pool and AC
- Recommended: ST001 (Riad Porte Royale), ST005 (Kasbah Mohayut), ST010 (Riad Taroudant)
- Secure parking flagged as mandatory selection criterion — properties without confirmed bike security excluded

---

### Rule Set D: Group Size Adjustments

**Rule D1 — Solo riders:**
- All routes eligible
- Flag isolation risk for RT003 (no group support on remote sections)
- Recommend social auberges (ST004, ST006) for solo atmosphere
- Cell coverage notes critical for emergency planning

**Rule D2 — Groups (3-8 riders):**
- Stay selection requires group_friendly = yes AND parking capacity ≥ group_size
- RT001 and RT002 are best group routes (wide paved roads, regular waypoints)
- Note: groups > 6 should book ALL accommodation minimum 2 weeks ahead in peak season

**Rule D3 — Large groups (9-15 riders):**
- Limit to RT001 and RT002 main routes
- Require dedicated overnight confirmation at each waypoint
- Flag that some auberges (ST003, ST006) cannot accept ≥8 bikes — route alternative accommodations
- Food stops should be confirmed in advance for groups > 8

---

### Rule Set E: Trip Duration Optimization

**Rule E1 — Short trips (3-4 days):**
- Recommend RT002 (Road of Kasbahs — 2 days) or RT004 partial (Fès to Merzouga — 2 days)
- RT001 in compressed 4-day version possible but tiring; flag pacing risk

**Rule E2 — Standard trips (5-7 days):**
- RT001 full loop (7 days preferred, 5 days compressed)
- RT003 + RT002 combo (5 days: Marrakech to Agadir via Test, north through Souss)

**Rule E3 — Extended trips (8+ days):**
- Full RT001 with Rif extension (add RT006 first or last 3-4 days)
- Allow detours to Msemrir, M'Hamid, Atlantic coast

---

### Rule Set F: Seasonal Safety Overrides

**Rule F1 — Winter travel (December–February):**
- Automatically exclude routes with passes above 1800m from recommendation
- Flag RT001 Tichka crossing: status check required
- Flag RT003 Tizi n'Test: strongly advise against (not just flag)
- Redirect to Draa Valley lower route (via N9) as Tichka alternative in snow season
- Desert routes (RT004, Merzouga sector) elevated in priority in winter — pleasant conditions

**Rule F2 — Summer travel (June–August):**
- Flag all desert routes with extreme heat warning
- Recommend dawn-to-10am and 4pm-sunset riding windows only
- Require hydration planning notes in itinerary
- Coastal route (RT005 coast section) promoted in summer as heat-relief option

---

# SECTION 9: RECOMMENDATION OUTPUT LOGIC

## AI Recommendation Generation Process

### Step 1: Profile Parsing and Route Eligibility
The system begins by parsing the rider profile to extract key decision variables:
- experience_level → filters routes by difficulty
- bike_type → filters routes by surface compatibility
- trip_duration → constrains total circuit distance
- preferred_vibe → ranks routes by character match
- budget → pre-selects compatible stays and restaurants
- travel_season → applies weather risk filters (Section 7) and seasonal overrides (Rule Set F)

Routes are ranked by a composite compatibility score computed from these variables. If multiple routes score equally, the route with higher scenic_score for the declared vibe preference is selected.

---

### Step 2: GPX-Based Stop Sequence Construction
Once the primary route is selected, the system loads the linked GPX file (linked_gpx_name field) and the route's segment table (Section 3). Segments are loaded in order (segment_order) and analyzed for:
- stop_recommendation flags → mandatory, optional, or bypass
- nearest_place_id → matched to places database for context
- nearest_fuel_station_id → fuel gap logic applied using Rule Set B3 and bike tank capacity
- nearest_stay_id → filtered by budget and group size

The system inserts recommended stops into the itinerary at logical intervals, ensuring:
- No fuel gap exceeds the bike's estimated range (tank_size × 0.85 for reserve)
- Daily mileage does not exceed the rider's daily_ride_km_tolerance
- Overnight stops are at places with confirmed motorbike_friendly = yes AND secure_parking = yes (or explicitly accepted by comfort-tolerant riders)

---

### Step 3: Overnight Stop Selection Logic
For each recommended overnight node, the system:
1. Filters stays by: place_id match, price_range within budget, secure_parking where required, group_friendly where group_size > 3
2. Ranks filtered stays by: motorbike_friendly score, comfort_score (matched to sleep_preference), then user rating
3. Selects primary recommendation + one alternative
4. Flags booking requirement: if best_season = current travel month AND stay appears in multiple route recommendations → mark as "book in advance"

---

### Step 4: Food Stop Insertion
Food stops are inserted at logical midday points within each day's segment sequence:
- Prioritize stops where stop_recommendation includes lunch
- Match biker_friendly = yes AND quick_stop_friendly = yes for tight itineraries
- For scenic detours or slower paced days: prefer stops with scenic vibe and longer meal experience
- Group riders: prefer group_friendly restaurants with pre-booking notes for groups >6

---

### Step 5: Fuel Planning Integration
The system calculates the cumulative distance between confirmed fuel stations along the route, using nearest_fuel_station_id references in segments:
- If gap exceeds 85% of estimated bike range → insert mandatory fuel warning at previous station
- If gap exceeds 100% of estimated range → flag route as incompatible with that bike type OR suggest 2L portable fuel carrier
- For scooters and small-tank bikes → tighten threshold to 70% of estimated range
- Critical desert sections (Erfoud to Merzouga, Asni to Ouled Berhil) always flagged with mandatory fuel discipline note

---

### Step 6: Weather and Safety Warning Integration
Before outputting the final itinerary, the system queries the Weather Risks table (Section 7) for:
- All risk entries where related_route_id matches the selected route
- All risk entries where related_place_id matches any place in the itinerary
- All risk entries where season matches the travel_season input

Triggered warnings are sorted by severity:
- very high → inserted as red-flag alerts at the relevant itinerary step
- high → inserted as prominent caution notes
- medium → inserted as background notes in the day summary

Seasonal pass closures (WR001, WR002) trigger alternative routing logic if travel_season = Nov-Mar.

---

### Step 7: Final Itinerary Output Structure
The recommendation is output as a structured day-by-day itinerary with the following components per day:

**Day header:** Day number, start place, end place, total distance, estimated ride time, difficulty summary

**Segment narrative:** A paragraph describing the riding character of each segment — road feel, scenery, what to watch for — written in a voice appropriate to the rider's experience level

**Recommended stops (ordered):** Morning departure note → fuel stop(s) with station name and distance → lunch recommendation with restaurant name and brief context → afternoon riding note → overnight recommendation with stay name, price range, and booking note

**Weather alert block:** Any triggered weather risks from the risk database, with rider_advice text

**Why this fits you:** A personalized closing paragraph that explicitly references the rider's profile elements (experience, bike, vibe preference, budget) and explains why each day's plan matches them — this is the critical explainability layer of the recommendation engine

---

### Example Logic Trace

**Input profile:**
- Experience: intermediate
- Bike: BMW F850GS
- Group: solo
- Budget: €50/day
- Trip: 7 days
- Preferred vibe: desert + adventure
- Sleep: mid-range auberge
- Season: October

**System output logic:**
1. Route filter → RT001 (Grand Atlas Circuit) scores highest: intermediate-compatible, adventure bike suited, desert + mountain scenery, 7-day duration match
2. Season check → October: no pass closures, excellent desert conditions, book Merzouga in advance
3. GPX load → RT001 segment sequence loaded, stops inserted at 100–180km daily intervals
4. Fuel check → Erfoud fill mandatory flagged (GS003), Asni alternative bypass (not on RT001)
5. Stay selection → mid-range filter: ST002 (Ouarzazate), ST008 (Dadès Gorge), ST005 (Merzouga), ST011 (Midelt) — all within €40–60 range
6. Food selection → RS002 (Ouarzazate dinner), RS004 (Boumalne lunch), RS009 (Merzouga sunset dinner)
7. Weather check → WR010 (Tichka crosswind — medium, October), no severe triggers
8. Output → 7-day itinerary with personalized narrative, fuel warnings, Merzouga booking flag, and crosswind note on Day 2

---

# APPENDIX A: QUICK REFERENCE — CRITICAL FUEL GAPS

| gap_id | from_station | to_station | distance_km | risk_level | notes |
|--------|-------------|------------|-------------|------------|-------|
| FG001 | GS006 (Asni) | GS007 (Ouled Berhil) | ~200 | CRITICAL | Tizi n'Test crossing — longest paved fuel gap in Morocco |
| FG002 | GS003 (Erfoud) | Merzouga village | 53 | HIGH | Merzouga station unreliable — always fill at Erfoud |
| FG003 | GS002 (Ouarzazate) | GS004 (Tinerhir) | ~120 | MEDIUM | Multiple small towns but no guaranteed open station between |
| FG004 | GS001 (Marrakech) | GS002 (Ouarzazate) | ~200 | MEDIUM | Aït Ourir has stations but limited stock; fill fully in Marrakech |

---

# APPENDIX B: SEASONAL CALENDAR

| month | recommended_regions | avoid_regions | key_notes |
|-------|--------------------|--------------|----|
| Jan | Draa Valley, Merzouga (desert) | All passes above 1800m | Desert excellent; passes risky |
| Feb | Desert (Merzouga peak), Draa | High Atlas passes | Best dune photography light |
| Mar | Desert, Kasbahs Road, Taroudant | Tizi n'Test (ice risk) | Wildflowers begin; flash flood risk starts |
| Apr | All south circuits, Taroudant | None critical | Excellent all-round month |
| May | All Morocco circuits | Extreme south deserts (hot) | Rose harvest in Dadès/Skoura |
| Jun | Coastal, Rif, Chefchaouen | Desert zones, high passes (heat) | Atlantic coast pleasant |
| Jul | Rif, Chefchaouen, northern Morocco | All desert routes | Sahara dangerous |
| Aug | Rif, Atlantic coast only | All desert and inland routes | Peak heat inland |
| Sep | All Morocco circuits | None critical | Post-heat cooling; excellent conditions |
| Oct | All Morocco — BEST MONTH | None | Perfect temperatures everywhere |
| Nov | All south circuits | High Atlas passes (weather risk) | Check Tichka before each day |
| Dec | Desert (cool, clear) | All passes, Rif (rain) | Excellent desert; dangerous passes |

---

*End of Morocco Motorcycle Travel Knowledge Base v1.0*
*Structured for RAG indexing, vector embedding, JSON export, and GPX-linked route planning.*
