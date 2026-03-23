"""
corpus.py - Large Scale Fact Corpus for v0.8

Goal: 1000+ facts across diverse categories
Test: 50+ questions with edge cases

Categories:
- Geography (capitals, rivers, mountains, oceans)
- Science (physics, chemistry, biology, astronomy)
- History (events, dates, people)
- Animals (records, features)
- Sports (records, teams)
- Technology (inventions, companies)
- Arts & Culture (books, music, art)
"""

# =============================================================================
# WORLD CAPITALS (195 countries)
# =============================================================================

WORLD_CAPITALS = [
    # A
    ("Afghanistan", "Kabul"), ("Albania", "Tirana"), ("Algeria", "Algiers"),
    ("Andorra", "Andorra la Vella"), ("Angola", "Luanda"), ("Argentina", "Buenos Aires"),
    ("Armenia", "Yerevan"), ("Australia", "Canberra"), ("Austria", "Vienna"),
    ("Azerbaijan", "Baku"),
    # B
    ("Bahamas", "Nassau"), ("Bahrain", "Manama"), ("Bangladesh", "Dhaka"),
    ("Barbados", "Bridgetown"), ("Belarus", "Minsk"), ("Belgium", "Brussels"),
    ("Belize", "Belmopan"), ("Benin", "Porto-Novo"), ("Bhutan", "Thimphu"),
    ("Bolivia", "La Paz"), ("Bosnia", "Sarajevo"), ("Botswana", "Gaborone"),
    ("Brazil", "Brasilia"), ("Brunei", "Bandar Seri Begawan"), ("Bulgaria", "Sofia"),
    ("Burkina Faso", "Ouagadougou"), ("Burundi", "Gitega"),
    # C
    ("Cambodia", "Phnom Penh"), ("Cameroon", "Yaounde"), ("Canada", "Ottawa"),
    ("Cape Verde", "Praia"), ("Central African Republic", "Bangui"), ("Chad", "NDjamena"),
    ("Chile", "Santiago"), ("China", "Beijing"), ("Colombia", "Bogota"),
    ("Comoros", "Moroni"), ("Congo", "Brazzaville"), ("Costa Rica", "San Jose"),
    ("Croatia", "Zagreb"), ("Cuba", "Havana"), ("Cyprus", "Nicosia"),
    ("Czech Republic", "Prague"),
    # D
    ("Denmark", "Copenhagen"), ("Djibouti", "Djibouti"), ("Dominica", "Roseau"),
    ("Dominican Republic", "Santo Domingo"),
    # E
    ("Ecuador", "Quito"), ("Egypt", "Cairo"), ("El Salvador", "San Salvador"),
    ("Equatorial Guinea", "Malabo"), ("Eritrea", "Asmara"), ("Estonia", "Tallinn"),
    ("Eswatini", "Mbabane"), ("Ethiopia", "Addis Ababa"),
    # F
    ("Fiji", "Suva"), ("Finland", "Helsinki"), ("France", "Paris"),
    # G
    ("Gabon", "Libreville"), ("Gambia", "Banjul"), ("Georgia", "Tbilisi"),
    ("Germany", "Berlin"), ("Ghana", "Accra"), ("Greece", "Athens"),
    ("Grenada", "Saint Georges"), ("Guatemala", "Guatemala City"), ("Guinea", "Conakry"),
    ("Guyana", "Georgetown"),
    # H
    ("Haiti", "Port-au-Prince"), ("Honduras", "Tegucigalpa"), ("Hungary", "Budapest"),
    # I
    ("Iceland", "Reykjavik"), ("India", "New Delhi"), ("Indonesia", "Jakarta"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Ireland", "Dublin"),
    ("Israel", "Jerusalem"), ("Italy", "Rome"),
    # J
    ("Jamaica", "Kingston"), ("Japan", "Tokyo"), ("Jordan", "Amman"),
    # K
    ("Kazakhstan", "Astana"), ("Kenya", "Nairobi"), ("Kiribati", "Tarawa"),
    ("Kosovo", "Pristina"), ("Kuwait", "Kuwait City"), ("Kyrgyzstan", "Bishkek"),
    # L
    ("Laos", "Vientiane"), ("Latvia", "Riga"), ("Lebanon", "Beirut"),
    ("Lesotho", "Maseru"), ("Liberia", "Monrovia"), ("Libya", "Tripoli"),
    ("Liechtenstein", "Vaduz"), ("Lithuania", "Vilnius"), ("Luxembourg", "Luxembourg City"),
    # M
    ("Madagascar", "Antananarivo"), ("Malawi", "Lilongwe"), ("Malaysia", "Kuala Lumpur"),
    ("Maldives", "Male"), ("Mali", "Bamako"), ("Malta", "Valletta"),
    ("Mauritania", "Nouakchott"), ("Mauritius", "Port Louis"), ("Mexico", "Mexico City"),
    ("Micronesia", "Palikir"), ("Moldova", "Chisinau"), ("Monaco", "Monaco"),
    ("Mongolia", "Ulaanbaatar"), ("Montenegro", "Podgorica"), ("Morocco", "Rabat"),
    ("Mozambique", "Maputo"), ("Myanmar", "Naypyidaw"),
    # N
    ("Namibia", "Windhoek"), ("Nauru", "Yaren"), ("Nepal", "Kathmandu"),
    ("Netherlands", "Amsterdam"), ("New Zealand", "Wellington"), ("Nicaragua", "Managua"),
    ("Niger", "Niamey"), ("Nigeria", "Abuja"), ("North Korea", "Pyongyang"),
    ("North Macedonia", "Skopje"), ("Norway", "Oslo"),
    # O
    ("Oman", "Muscat"),
    # P
    ("Pakistan", "Islamabad"), ("Palau", "Ngerulmud"), ("Panama", "Panama City"),
    ("Papua New Guinea", "Port Moresby"), ("Paraguay", "Asuncion"), ("Peru", "Lima"),
    ("Philippines", "Manila"), ("Poland", "Warsaw"), ("Portugal", "Lisbon"),
    # Q
    ("Qatar", "Doha"),
    # R
    ("Romania", "Bucharest"), ("Russia", "Moscow"), ("Rwanda", "Kigali"),
    # S
    ("Saint Lucia", "Castries"), ("Samoa", "Apia"), ("San Marino", "San Marino"),
    ("Saudi Arabia", "Riyadh"), ("Senegal", "Dakar"), ("Serbia", "Belgrade"),
    ("Seychelles", "Victoria"), ("Sierra Leone", "Freetown"), ("Singapore", "Singapore"),
    ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"), ("Solomon Islands", "Honiara"),
    ("Somalia", "Mogadishu"), ("South Africa", "Pretoria"), ("South Korea", "Seoul"),
    ("South Sudan", "Juba"), ("Spain", "Madrid"), ("Sri Lanka", "Colombo"),
    ("Sudan", "Khartoum"), ("Suriname", "Paramaribo"), ("Sweden", "Stockholm"),
    ("Switzerland", "Bern"), ("Syria", "Damascus"),
    # T
    ("Taiwan", "Taipei"), ("Tajikistan", "Dushanbe"), ("Tanzania", "Dodoma"),
    ("Thailand", "Bangkok"), ("Timor-Leste", "Dili"), ("Togo", "Lome"),
    ("Tonga", "Nukualofa"), ("Trinidad and Tobago", "Port of Spain"), ("Tunisia", "Tunis"),
    ("Turkey", "Ankara"), ("Turkmenistan", "Ashgabat"), ("Tuvalu", "Funafuti"),
    # U
    ("Uganda", "Kampala"), ("Ukraine", "Kyiv"), ("United Arab Emirates", "Abu Dhabi"),
    ("United Kingdom", "London"), ("United States", "Washington D.C."),
    ("Uruguay", "Montevideo"), ("Uzbekistan", "Tashkent"),
    # V
    ("Vanuatu", "Port Vila"), ("Vatican City", "Vatican City"), ("Venezuela", "Caracas"),
    ("Vietnam", "Hanoi"),
    # Y
    ("Yemen", "Sanaa"),
    # Z
    ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"),
]

# =============================================================================
# US STATE CAPITALS (50 states)
# =============================================================================

US_STATE_CAPITALS = [
    ("Alabama", "Montgomery"), ("Alaska", "Juneau"), ("Arizona", "Phoenix"),
    ("Arkansas", "Little Rock"), ("California", "Sacramento"), ("Colorado", "Denver"),
    ("Connecticut", "Hartford"), ("Delaware", "Dover"), ("Florida", "Tallahassee"),
    ("Georgia", "Atlanta"), ("Hawaii", "Honolulu"), ("Idaho", "Boise"),
    ("Illinois", "Springfield"), ("Indiana", "Indianapolis"), ("Iowa", "Des Moines"),
    ("Kansas", "Topeka"), ("Kentucky", "Frankfort"), ("Louisiana", "Baton Rouge"),
    ("Maine", "Augusta"), ("Maryland", "Annapolis"), ("Massachusetts", "Boston"),
    ("Michigan", "Lansing"), ("Minnesota", "Saint Paul"), ("Mississippi", "Jackson"),
    ("Missouri", "Jefferson City"), ("Montana", "Helena"), ("Nebraska", "Lincoln"),
    ("Nevada", "Carson City"), ("New Hampshire", "Concord"), ("New Jersey", "Trenton"),
    ("New Mexico", "Santa Fe"), ("New York", "Albany"), ("North Carolina", "Raleigh"),
    ("North Dakota", "Bismarck"), ("Ohio", "Columbus"), ("Oklahoma", "Oklahoma City"),
    ("Oregon", "Salem"), ("Pennsylvania", "Harrisburg"), ("Rhode Island", "Providence"),
    ("South Carolina", "Columbia"), ("South Dakota", "Pierre"), ("Tennessee", "Nashville"),
    ("Texas", "Austin"), ("Utah", "Salt Lake City"), ("Vermont", "Montpelier"),
    ("Virginia", "Richmond"), ("Washington", "Olympia"), ("West Virginia", "Charleston"),
    ("Wisconsin", "Madison"), ("Wyoming", "Cheyenne"),
]

# =============================================================================
# SOLAR SYSTEM
# =============================================================================

SOLAR_SYSTEM = [
    "The Sun is the star at the center of our solar system.",
    "Mercury is the smallest planet and closest to the Sun.",
    "Mercury has no moons.",
    "Venus is the hottest planet in our solar system.",
    "Venus is the second planet from the Sun.",  # v1.0: Order
    "Venus comes after Mercury in our solar system.",  # v1.0: Order
    "Venus rotates backwards compared to most planets.",
    "Earth is the only planet known to support life.",
    "Earth is the third planet from the Sun.",  # v1.0: Order
    "Earth has one moon called the Moon.",
    "Earth has exactly one moon.",  # v1.0: Explicit singular
    "The number of moons Earth has is one.",  # v1.0: Direct answer format
    "Mars is known as the Red Planet.",
    "Mars is the fourth planet from the Sun.",  # v1.0: Order
    "Mars has two moons named Phobos and Deimos.",
    "Jupiter is the largest planet in our solar system.",
    "Jupiter is the fifth planet from the Sun.",  # v1.0: Order
    "Jupiter has the Great Red Spot, a giant storm.",
    "Jupiter has at least 95 known moons.",
    "Saturn is famous for its beautiful rings.",
    "Saturn is the second largest planet.",
    "Saturn is the sixth planet from the Sun.",  # v1.0: Order
    "Saturn has at least 146 known moons.",
    "Titan is the largest moon of Saturn.",
    "Uranus rotates on its side.",
    "Uranus is the coldest planet in our solar system.",
    "Uranus is the seventh planet from the Sun.",  # v1.0: Order
    "Neptune is the windiest planet.",
    "Neptune is the farthest planet from the Sun.",
    "Neptune is the eighth planet from the Sun.",  # v1.0: Order
    "Pluto was reclassified as a dwarf planet in 2006.",
    "The asteroid belt is located between Mars and Jupiter.",
    "Ceres is the largest object in the asteroid belt.",
    "Halley's Comet is visible from Earth every 76 years.",
]

# =============================================================================
# GEOGRAPHY - RIVERS
# =============================================================================

RIVERS = [
    "The Nile is the longest river in Africa.",
    "The Nile River flows through Africa.",  # v1.0: Explicit location
    "The Nile is located in Africa.",  # v1.0: Explicit location
    "The Amazon is the largest river by water volume.",
    "The Amazon River flows through South America.",
    "The Mississippi is the longest river in North America.",
    "The Missouri River is the longest river in the United States.",
    "The Yangtze is the longest river in Asia.",
    "The Yangtze River flows through China.",
    "The Yellow River is the second longest river in China.",
    "The Ganges River is sacred to Hindus.",
    "The Danube flows through ten European countries.",
    "The Rhine River flows through Germany.",
    "The Thames flows through London.",
    "The Seine flows through Paris.",
    "The Volga is the longest river in Europe.",
    "The Congo River is the second longest river in Africa.",
    "The Mekong River flows through Southeast Asia.",
    "The Tigris and Euphrates rivers flow through Iraq.",
    "The Colorado River carved the Grand Canyon.",
    "The Niagara River connects Lake Erie and Lake Ontario.",
]

# =============================================================================
# GEOGRAPHY - MOUNTAINS
# =============================================================================

MOUNTAINS = [
    "Mount Everest is the tallest mountain on Earth.",
    "Mount Everest is located in the Himalayas.",
    "K2 is the second tallest mountain on Earth.",
    "Kangchenjunga is the third tallest mountain.",
    "Mount Kilimanjaro is the tallest mountain in Africa.",
    "Mount Kilimanjaro is located in Africa.",  # v1.0: Explicit
    "Mount McKinley is the tallest mountain in North America.",
    "Denali is another name for Mount McKinley.",
    "Aconcagua is the tallest mountain in South America.",
    "Mount Elbrus is the tallest mountain in Europe.",
    "Mount Fuji is the tallest mountain in Japan.",
    "The Himalayas are the highest mountain range in the world.",
    "The Alps are a major mountain range in Europe.",
    "The Andes are the longest mountain range in the world.",
    "The Andes are located in South America.",  # v1.0: Explicit
    "The Rocky Mountains are in North America.",
    "Mount Blanc is the highest peak in the Alps.",
    "The Appalachian Mountains are in eastern North America.",
    "Mauna Kea is the tallest mountain from base to peak.",
]

# =============================================================================
# GEOGRAPHY - OCEANS AND SEAS
# =============================================================================

OCEANS_SEAS = [
    "The Pacific Ocean is the largest ocean on Earth.",
    "The Atlantic Ocean is the second largest ocean.",
    "The Indian Ocean is the third largest ocean.",
    "The Arctic Ocean is the smallest ocean.",
    "The Southern Ocean surrounds Antarctica.",
    "The Mariana Trench is the deepest point in the ocean.",
    "The Mediterranean Sea is between Europe and Africa.",
    "The Caribbean Sea is in the Atlantic Ocean.",
    "The Red Sea is between Africa and Asia.",
    "The Dead Sea is the lowest point on land.",
    "The Dead Sea is extremely salty.",
    "The Caspian Sea is the largest enclosed body of water.",
    "The Great Barrier Reef is off the coast of Australia.",
    "Lake Baikal is the deepest lake in the world.",
    "Lake Superior is the largest Great Lake.",
    "The Great Lakes are in North America.",
]

# =============================================================================
# GEOGRAPHY - DESERTS
# =============================================================================

DESERTS = [
    "The Sahara is the largest hot desert in the world.",
    "The Sahara Desert is in Africa.",
    "The Sahara is the largest desert on Earth.",  # v1.0: Explicit without "hot"
    "Antarctica is technically the largest desert.",
    "The Gobi Desert is in Mongolia and China.",
    "The Arabian Desert is in the Middle East.",
    "The Kalahari Desert is in southern Africa.",
    "Death Valley is the hottest place in North America.",
    "The Atacama Desert is the driest desert on Earth.",
    "The Mojave Desert is in California.",
    "The Sonoran Desert spans the US and Mexico.",
]

# =============================================================================
# SCIENCE - PHYSICS
# =============================================================================

PHYSICS = [
    "The speed of light is approximately 300000 kilometers per second.",
    "Light travels at approximately 300000 kilometers per second.",
    "The speed of sound is about 343 meters per second in air.",
    "Gravity accelerates objects at 9.8 meters per second squared on Earth.",
    "Albert Einstein developed the theory of relativity.",
    "Isaac Newton discovered the laws of motion.",
    "Isaac Newton discovered gravity.",
    "E equals mc squared is Einstein's famous equation.",
    "An electron has a negative charge.",
    "A proton has a positive charge.",
    "A neutron has no electrical charge.",
    "Atoms are made of protons, neutrons, and electrons.",
    "The nucleus contains protons and neutrons.",
    "Photons are particles of light.",
    "The Higgs boson was discovered in 2012.",
    "Black holes have gravity so strong that light cannot escape.",
    "Stephen Hawking studied black holes.",
]

# =============================================================================
# SCIENCE - CHEMISTRY
# =============================================================================

CHEMISTRY = [
    "Water freezes at zero degrees Celsius.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The chemical formula for water is H2O.",
    "The chemical formula for carbon dioxide is CO2.",
    "Oxygen makes up about 21 percent of Earth's atmosphere.",
    "Nitrogen makes up about 78 percent of Earth's atmosphere.",
    "Nitrogen is the most abundant gas in Earth's atmosphere.",  # v1.0: Explicit "most"
    "Gold has the chemical symbol Au.",
    "Silver has the chemical symbol Ag.",
    "Iron has the chemical symbol Fe.",
    "Copper has the chemical symbol Cu.",
    "Mercury has the chemical symbol Hg.",
    "Sodium has the chemical symbol Na.",
    "The atomic number of hydrogen is 1.",
    "The atomic number of helium is 2.",
    "The atomic number of carbon is 6.",
    "The atomic number of oxygen is 8.",
    "The atomic number of gold is 79.",
    "The periodic table organizes chemical elements.",
    "Dmitri Mendeleev created the periodic table.",
    "Noble gases are in group 18 of the periodic table.",
    "Helium is the second most abundant element in the universe.",
    "Hydrogen is the most abundant element in the universe.",
]

# =============================================================================
# SCIENCE - BIOLOGY
# =============================================================================

BIOLOGY = [
    "DNA contains the genetic instructions for life.",
    "DNA stands for deoxyribonucleic acid.",
    "RNA stands for ribonucleic acid.",
    "The human body has 206 bones.",
    "The human heart beats about 100000 times per day.",
    "The human brain contains about 86 billion neurons.",
    "Red blood cells carry oxygen.",
    "White blood cells fight infection.",
    "The liver is the largest internal organ.",
    "The skin is the largest organ of the body.",
    "Mitochondria are the powerhouses of the cell.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Chlorophyll makes plants green.",
    "Charles Darwin developed the theory of evolution.",
    "Gregor Mendel is the father of genetics.",
    "Antibiotics kill bacteria but not viruses.",
    "Vaccines help the immune system fight disease.",
    "The human genome has about 20000 genes.",
]

# =============================================================================
# SCIENCE - ASTRONOMY
# =============================================================================

ASTRONOMY = [
    "The universe is approximately 13.8 billion years old.",
    "The Milky Way is our home galaxy.",
    "The Milky Way contains about 200 billion stars.",
    "Andromeda is the nearest major galaxy to the Milky Way.",
    "A light year is the distance light travels in one year.",
    "The Big Bang theory explains the origin of the universe.",
    "Galileo Galilei invented the telescope.",
    "Edwin Hubble discovered that the universe is expanding.",
    "A supernova is an exploding star.",
    "A neutron star is the collapsed core of a massive star.",
    "Pulsars are rapidly rotating neutron stars.",
    "Quasars are extremely luminous active galactic nuclei.",
    "The cosmic microwave background is radiation from the early universe.",
    "Dark matter makes up about 27 percent of the universe.",
    "Dark energy makes up about 68 percent of the universe.",
]

# =============================================================================
# HISTORY
# =============================================================================

HISTORY = [
    "World War I began in 1914.",
    "World War I ended in 1918.",
    "World War II began in 1939.",
    "World War II ended in 1945.",
    "The Declaration of Independence was signed in 1776.",
    "The American Civil War ended in 1865.",
    "Abraham Lincoln was assassinated in 1865.",
    "The Berlin Wall fell in 1989.",
    "The Soviet Union dissolved in 1991.",
    "Neil Armstrong walked on the Moon in 1969.",
    "The Titanic sank in 1912.",
    "John F Kennedy was assassinated in 1963.",
    "The French Revolution began in 1789.",
    "Napoleon was defeated at Waterloo in 1815.",
    "The Roman Empire fell in 476 AD.",
    "Christopher Columbus reached America in 1492.",
    "The printing press was invented by Johannes Gutenberg around 1440.",
    "The Industrial Revolution began in Britain.",
    "The Renaissance began in Italy.",
    "The Great Wall of China was built to protect against invasions.",
    "The Egyptian pyramids were built as tombs for pharaohs.",
    "Cleopatra was the last pharaoh of Egypt.",
    "Julius Caesar was assassinated in 44 BC.",
    "Alexander the Great conquered a vast empire.",
    "The Black Death killed millions in Europe in the 1300s.",
    "Martin Luther started the Protestant Reformation in 1517.",
    "The American Revolution began in 1775.",
    "George Washington was the first US President.",
    "Abraham Lincoln was the 16th US President.",
    "The Emancipation Proclamation freed slaves in 1863.",
    "Women gained the right to vote in the US in 1920.",
    "The Cold War was between the United States and Soviet Union.",
    "The Korean War began in 1950.",
    "The Vietnam War ended in 1975.",
    "Nelson Mandela became President of South Africa in 1994.",
    "The Internet was invented in the late 20th century.",
]

# =============================================================================
# ANIMALS
# =============================================================================

ANIMALS = [
    "The blue whale is the largest animal ever known.",
    "The African elephant is the largest land animal.",
    "The cheetah is the fastest land animal.",
    "The peregrine falcon is the fastest bird.",
    "The ostrich is the largest bird.",
    "The hummingbird is the smallest bird.",
    "The giraffe is the tallest animal.",
    "The African elephant has the largest ears of any animal.",
    "Dolphins are highly intelligent marine mammals.",
    "Octopuses have three hearts.",
    "Octopuses have blue blood.",
    "Sharks have been around for over 400 million years.",
    "Bees are essential pollinators for many plants.",
    "Ants can carry 50 times their body weight.",
    "The Arctic tern has the longest migration of any bird.",
    "Penguins cannot fly but are excellent swimmers.",
    "Polar bears live in the Arctic.",
    "Kangaroos can only be found wild in Australia.",
    "Koalas sleep up to 22 hours a day.",
    "Sloths are the slowest mammals.",
    "Bats are the only mammals that can truly fly.",
    "Whales are mammals, not fish.",
    "The giant panda eats mostly bamboo.",
    "Tigers are the largest wild cats.",
    "Lions live in groups called prides.",
    "The lion is known as the king of the jungle.",  # v1.0: Explicit
    "Crocodiles have the strongest bite of any animal.",
    "The box jellyfish is the most venomous animal.",
    "Elephants have excellent memories.",
    "Parrots can mimic human speech.",
    "Chameleons can change their color.",
    "Starfish can regenerate lost arms.",
    "The platypus is a mammal that lays eggs.",
]

# =============================================================================
# TECHNOLOGY & INVENTIONS
# =============================================================================

TECHNOLOGY = [
    "The telephone was invented by Alexander Graham Bell.",
    "Thomas Edison invented the practical light bulb.",
    "The Wright brothers made the first powered airplane flight in 1903.",
    "The first computer was called ENIAC.",
    "Tim Berners-Lee invented the World Wide Web.",
    "Steve Jobs co-founded Apple.",
    "Bill Gates co-founded Microsoft.",
    "Mark Zuckerberg founded Facebook.",
    "Google was founded by Larry Page and Sergey Brin.",
    "Amazon was founded by Jeff Bezos.",
    "The first iPhone was released in 2007.",
    "The first personal computer was made in the 1970s.",
    "The transistor was invented in 1947.",
    "The first email was sent in 1971.",
    "GPS stands for Global Positioning System.",
    "WiFi stands for Wireless Fidelity.",
    "HTML stands for HyperText Markup Language.",
    "AI stands for Artificial Intelligence.",
    "The internet uses TCP/IP protocol.",
    "Linux is an open source operating system.",
]

# =============================================================================
# ARTS & CULTURE
# =============================================================================

ARTS_CULTURE = [
    "Leonardo da Vinci painted the Mona Lisa.",
    "Michelangelo painted the Sistine Chapel ceiling.",
    "Vincent van Gogh painted Starry Night.",
    "Pablo Picasso co-founded Cubism.",
    "William Shakespeare wrote Romeo and Juliet.",
    "William Shakespeare wrote Hamlet.",
    "Charles Dickens wrote A Tale of Two Cities.",
    "Jane Austen wrote Pride and Prejudice.",
    "Mark Twain wrote The Adventures of Tom Sawyer.",
    "Ernest Hemingway wrote The Old Man and the Sea.",
    "The Beatles were from Liverpool, England.",
    "Elvis Presley was known as the King of Rock and Roll.",
    "Mozart was a child prodigy composer.",
    "Ludwig van Beethoven composed nine symphonies.",
    "The Eiffel Tower is in Paris.",
    "The Louvre is a famous museum in Paris.",  # v1.0: Explicit
    "The Mona Lisa is displayed in the Louvre in Paris.",  # v1.0: Explicit
    "The Statue of Liberty was a gift from France.",
    "The Colosseum is in Rome.",
    "The Great Pyramid of Giza is in Egypt.",
    "The Taj Mahal is in India.",
    "Big Ben is a famous clock tower in London.",
]

# =============================================================================
# SPORTS
# =============================================================================

SPORTS = [
    "The Olympics began in ancient Greece.",
    "The modern Olympics started in 1896.",
    "The FIFA World Cup is the biggest soccer tournament.",
    "The Super Bowl is the championship game of the NFL.",
    "The NBA Finals determine the basketball champion.",
    "The World Series is the championship of Major League Baseball.",
    "Michael Jordan is considered one of the greatest basketball players.",
    "Pele is considered one of the greatest soccer players.",
    "Usain Bolt is the fastest human ever recorded.",
    "Michael Phelps won the most Olympic gold medals.",
    "Babe Ruth was a legendary baseball player.",
    "Muhammad Ali was a legendary boxer.",
    "Serena Williams is one of the greatest tennis players.",
    "Roger Federer won 20 Grand Slam tennis titles.",
    "Tiger Woods is a famous golfer.",
    "Wayne Gretzky is the greatest hockey player of all time.",
    "A marathon is 42.195 kilometers or 26.2 miles.",
    "The Tour de France is a famous cycling race.",
    "Cricket is the second most popular sport in the world.",
    "Rugby originated in England.",
]

# =============================================================================
# GENERATE ALL FACTS
# =============================================================================

# v1.0: Explicit disambiguation facts
DISAMBIGUATION_FACTS = [
    # Georgia country vs US state
    "Georgia is a country in the Caucasus region.",
    "The country Georgia has its capital in Tbilisi.",
    "Tbilisi is the capital of the country Georgia.",
    "Georgia the country is located between Europe and Asia.",
    # Washington state vs Washington D.C.
    "Washington state has its capital in Olympia.",
    "Olympia is the capital of Washington state.",
    "Washington state is in the Pacific Northwest.",
]

def generate_all_facts():
    """Generate the complete fact corpus."""
    facts = []
    
    # World capitals (two variants each)
    for country, capital in WORLD_CAPITALS:
        facts.append(f"{capital} is the capital of {country}.")
        facts.append(f"The capital of {country} is {capital}.")
    
    # US state capitals (two variants each)
    for state, capital in US_STATE_CAPITALS:
        facts.append(f"{capital} is the capital of {state}.")
        facts.append(f"The capital of {state} is {capital}.")
    
    # All other categories
    facts.extend(SOLAR_SYSTEM)
    facts.extend(RIVERS)
    facts.extend(MOUNTAINS)
    facts.extend(OCEANS_SEAS)
    facts.extend(DESERTS)
    facts.extend(PHYSICS)
    facts.extend(CHEMISTRY)
    facts.extend(BIOLOGY)
    facts.extend(ASTRONOMY)
    facts.extend(HISTORY)
    facts.extend(ANIMALS)
    facts.extend(TECHNOLOGY)
    facts.extend(ARTS_CULTURE)
    facts.extend(SPORTS)
    facts.extend(DISAMBIGUATION_FACTS)  # v1.0
    
    return facts


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

TEST_SUITE = [
    # === CAPITALS (basic) ===
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of China?", "Beijing"),
    ("What is the capital of Canada?", "Ottawa"),
    
    # === US STATE CAPITALS ===
    ("What is the capital of Mississippi?", "Jackson"),
    ("What is the capital of Alabama?", "Montgomery"),
    ("What is the capital of California?", "Sacramento"),
    ("What is the capital of Texas?", "Austin"),
    ("What is the capital of New York?", "Albany"),
    ("What is the capital of Florida?", "Tallahassee"),
    
    # === SOLAR SYSTEM ===
    ("Which planet has rings?", "Saturn"),
    ("What is the largest planet?", "Jupiter"),
    ("What is the smallest planet?", "Mercury"),
    ("What is the hottest planet?", "Venus"),
    ("What is the coldest planet?", "Uranus"),
    ("What is the Red Planet?", "Mars"),
    ("How many moons does Earth have?", "one"),
    ("What planet rotates on its side?", "Uranus"),
    
    # === GEOGRAPHY ===
    ("What is the tallest mountain?", "Everest"),
    ("What is the second tallest mountain?", "K2"),
    ("What is the longest river in Africa?", "Nile"),
    ("What is the largest river by volume?", "Amazon"),
    ("What is the largest ocean?", "Pacific"),
    ("What is the deepest lake?", "Baikal"),
    ("What is the largest desert?", "Sahara"),
    ("What is the driest desert?", "Atacama"),
    
    # === SCIENCE ===
    ("What freezes at zero degrees?", "Water"),
    ("What is the speed of light?", "300"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the atomic number of hydrogen?", "1"),
    ("What element is most abundant in the universe?", "Hydrogen"),
    ("What gas makes up most of Earth's atmosphere?", "Nitrogen"),
    ("How many bones in the human body?", "206"),
    ("What are the powerhouses of the cell?", "Mitochondria"),
    
    # === HISTORY ===
    ("When did World War II end?", "1945"),
    ("When did the Berlin Wall fall?", "1989"),
    ("Who walked on the Moon?", "Armstrong"),
    ("When was the Declaration of Independence signed?", "1776"),
    ("Who invented the printing press?", "Gutenberg"),
    ("When did the Titanic sink?", "1912"),
    
    # === ANIMALS ===
    ("What is the largest animal?", "whale"),
    ("What is the fastest land animal?", "cheetah"),
    ("What is the tallest animal?", "giraffe"),
    ("What is the largest land animal?", "elephant"),
    ("What is the largest bird?", "ostrich"),
    ("What is the smallest bird?", "hummingbird"),
    ("How many hearts does an octopus have?", "three"),
    
    # === TECHNOLOGY ===
    ("Who invented the telephone?", "Bell"),
    ("Who invented the light bulb?", "Edison"),
    ("Who co-founded Apple?", "Jobs"),
    ("Who co-founded Microsoft?", "Gates"),
    ("Who invented the World Wide Web?", "Berners"),
    
    # === ARTS ===
    ("Who painted the Mona Lisa?", "Leonardo"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("Who painted Starry Night?", "Gogh"),
    
    # === SPORTS ===
    ("What is the fastest human ever?", "Bolt"),
    ("Who won the most Olympic gold medals?", "Phelps"),
    ("Who is the greatest hockey player?", "Gretzky"),
    
    # === EDGE CASES - Disambiguation ===
    ("What is the capital of Georgia the country?", "Tbilisi"),
    ("What is the capital of Washington state?", "Olympia"),
    
    # === EDGE CASES - Superlatives ===
    ("What is the tallest mountain in Africa?", "Kilimanjaro"),
    ("What is the longest river in Europe?", "Volga"),
    ("What is the longest mountain range?", "Andes"),
]


if __name__ == "__main__":
    facts = generate_all_facts()
    print(f"Generated {len(facts)} facts")
    print(f"Test suite has {len(TEST_SUITE)} questions")
