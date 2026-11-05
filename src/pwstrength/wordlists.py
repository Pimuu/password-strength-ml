"""Hand-written word lists used by the generator, the labeler and the feature extractor.

IMPORTANT: Nothing in this file comes from a real leaked-credential dataset.
Every list was written by hand for this project. The "common password" list
holds widely known, generic weak passwords (keyboard rows, digit runs,
"password"-style words) of the kind that security-awareness material quotes.
The ordering is a hand-assigned *illustrative* rank. It is not a frequency
taken from any breach.

Each list is ordered from "most likely to be guessed first" to "least likely".
The labeler's guess estimator uses a word's position as its rank, the same way
frequency-ranked dictionaries are used in zxcvbn-style estimators.
"""

# ---------------------------------------------------------------------------
# Common weak passwords (hand-curated, illustrative rank order).
# ---------------------------------------------------------------------------
COMMON_PASSWORDS = [
    "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234",
    "111111", "1234567", "dragon", "123123", "baseball", "abc123", "football",
    "monkey", "letmein", "696969", "shadow", "master", "666666", "qwertyuiop",
    "123321", "mustang", "1234567890", "michael", "654321", "superman",
    "1qaz2wsx", "7777777", "121212", "000000", "qazwsx", "123qwe", "killer",
    "trustno1", "jordan", "jennifer", "zxcvbnm", "asdfgh", "hunter", "buster",
    "soccer", "harley", "batman", "andrew", "tigger", "sunshine", "iloveyou",
    "2000", "charlie", "robert", "thomas", "hockey", "ranger", "daniel",
    "starwars", "klaster", "112233", "george", "computer", "michelle",
    "jessica", "pepper", "1111", "zxcvbn", "555555", "11111111", "131313",
    "freedom", "777777", "pass", "maggie", "159753", "aaaaaa", "ginger",
    "princess", "joshua", "cheese", "amanda", "summer", "love", "ashley",
    "nicole", "chelsea", "biteme", "matthew", "access", "yankees", "987654321",
    "dallas", "austin", "thunder", "taylor", "matrix", "minecraft", "william",
    "corvette", "hello", "martin", "heather", "secret", "merlin", "diamond",
    "1234qwer", "gfhjkm", "hammer", "silver", "222222", "88888888", "anthony",
    "justin", "test", "bailey", "q1w2e3r4t5", "patrick", "internet", "scooter",
    "orange", "11111", "golfer", "cookie", "richard", "samantha", "bigdog",
    "guitar", "jackson", "whatever", "mickey", "chicken", "sparky", "snoopy",
    "maverick", "phoenix", "camaro", "peanut", "morgan", "welcome", "falcon",
    "cowboy", "ferrari", "samsung", "andrea", "smokey", "steelers", "joseph",
    "mercedes", "dakota", "arsenal", "eagles", "melissa", "boomer", "booboo",
    "spider", "nascar", "monster", "tigers", "yellow", "xxxxxx", "123123123",
    "gateway", "marina", "diablo", "bulldog", "qwer1234", "compaq", "purple",
    "hardcore", "banana", "junior", "hannah", "123654", "porsche", "lakers",
    "iceman", "money", "cowboys", "987654", "london", "tennis", "999999",
    "ncc1701", "coffee", "scooby", "0000", "miller", "boston", "q1w2e3r4",
    "fuckoff", "brandon", "yamaha", "chester", "mother", "forever", "johnny",
    "edward", "333333", "oliver", "redsox", "player", "nikita", "knight",
    "fender", "barney", "midnight", "please", "brandy", "chicago", "badboy",
    "admin", "admin123", "root", "toor", "changeme", "default", "guest",
    "passw0rd", "p@ssw0rd", "password1", "password123", "qwerty123",
    "welcome1", "letmein1", "abc12345", "abcd1234", "1q2w3e4r", "1q2w3e",
    "asdf1234", "zaq12wsx", "aa123456", "iloveyou1", "princess1", "monkey1",
]

# ---------------------------------------------------------------------------
# English words, roughly ordered by everyday frequency (hand-assigned).
# Short function words are left out because they would match almost anywhere.
# ---------------------------------------------------------------------------
ENGLISH_WORDS = [
    "love", "time", "life", "home", "world", "house", "money", "music", "game",
    "friend", "family", "water", "happy", "light", "heart", "night", "sun",
    "star", "moon", "baby", "girl", "boy", "man", "woman", "king", "queen",
    "angel", "dream", "magic", "power", "secret", "sweet", "blue", "red",
    "green", "black", "white", "gold", "silver", "pink", "purple", "orange",
    "yellow", "summer", "winter", "spring", "autumn", "flower", "rose",
    "tiger", "lion", "eagle", "dragon", "monkey", "horse", "dog", "cat",
    "puppy", "kitty", "bear", "wolf", "fish", "bird", "snake", "shark",
    "apple", "banana", "cherry", "lemon", "peach", "mango", "cookie", "candy",
    "sugar", "honey", "coffee", "pizza", "chocolate", "butter", "bread",
    "cheese", "chicken", "pepper", "ocean", "river", "forest", "mountain",
    "island", "beach", "sky", "rain", "snow", "storm", "thunder", "fire",
    "ice", "earth", "stone", "rock", "metal", "diamond", "crystal", "pearl",
    "school", "teacher", "student", "doctor", "nurse", "police", "soldier",
    "pirate", "ninja", "wizard", "knight", "hunter", "warrior", "master",
    "player", "winner", "lucky", "crazy", "funny", "cool", "super", "hero",
    "zombie", "ghost", "devil", "demon", "shadow", "dark", "death", "blood",
    "soul", "spirit", "faith", "hope", "peace", "freedom", "justice", "truth",
    "honor", "glory", "victory", "legend", "story", "book", "paper", "pencil",
    "letter", "computer", "internet", "phone", "mobile", "camera", "video",
    "movie", "picture", "guitar", "piano", "drum", "dance", "party", "travel",
    "summit", "journey", "rocket", "planet", "galaxy", "space", "future",
    "past", "memory", "mirror", "window", "door", "garden", "kitchen",
    "table", "chair", "bottle", "glass", "silk", "velvet", "cotton", "jacket",
    "shirt", "shoes", "hat", "ring", "crown", "castle", "tower", "bridge",
    "road", "street", "city", "country", "village", "station", "airport",
    "office", "market", "bank", "hotel", "church", "temple", "palace",
    "football", "soccer", "tennis", "hockey", "golf", "boxing", "racing",
    "runner", "swimmer", "rider", "driver", "pilot", "sailor", "captain",
    "general", "major", "doctor", "lawyer", "farmer", "baker", "painter",
    "singer", "writer", "reader", "thinker", "builder", "maker", "keeper",
    "monday", "friday", "sunday", "january", "april", "june", "july",
    "october", "december", "morning", "evening", "midnight", "today",
    "tomorrow", "forever", "always", "never", "maybe", "hello", "welcome",
    "goodbye", "please", "thanks", "sorry", "change", "access", "login",
    "admin", "user", "system", "server", "network", "secure", "safety",
    "private", "public", "simple", "easy", "hard", "strong", "weak", "fast",
    "slow", "big", "small", "little", "giant", "tiny", "young", "old", "new",
    "first", "last", "best", "better", "great", "grand", "royal", "noble",
    "brave", "wild", "free", "quiet", "silent", "loud", "bright", "shiny",
    "golden", "frozen", "broken", "hidden", "lost", "found", "open", "closed",
    "orbit", "comet", "meteor", "nebula", "quantum", "vector", "matrix",
    "pixel", "cyber", "robot", "laser", "radar", "sonic", "turbo", "nitro",
    "falcon", "phoenix", "raven", "cobra", "viper", "panther", "jaguar",
    "leopard", "falcon", "hawk", "owl", "fox", "rabbit", "turtle", "dolphin",
    "whale", "penguin", "koala", "panda", "zebra", "giraffe", "camel",
    "cactus", "maple", "willow", "cedar", "oak", "pine", "lotus", "tulip",
    "daisy", "violet", "jasmine", "lavender", "marble", "granite", "copper",
    "bronze", "iron", "steel", "carbon", "oxygen", "helium", "neon", "argon",
    "anchor", "harbor", "compass", "lantern", "candle", "blanket", "pillow",
    "basket", "bucket", "hammer", "ladder", "wrench", "needle", "thread",
    "button", "zipper", "pocket", "wallet", "ticket", "parcel", "package",
    "engine", "motor", "wheel", "tractor", "bicycle", "scooter", "subway",
    "canyon", "valley", "meadow", "prairie", "desert", "glacier", "volcano",
    "tornado", "breeze", "shadow", "echo", "whisper", "riddle", "puzzle",
    "banjo", "violin", "trumpet", "flute", "cello", "harp", "opera", "ballet",
]

# ---------------------------------------------------------------------------
# First names (generic, hand-picked).
# ---------------------------------------------------------------------------
NAMES = [
    "michael", "jennifer", "david", "jessica", "james", "ashley", "john",
    "sarah", "robert", "emily", "daniel", "amanda", "chris", "nicole", "mike",
    "anna", "alex", "maria", "thomas", "laura", "kevin", "lisa", "brian",
    "emma", "jason", "sophia", "ryan", "olivia", "eric", "hannah", "mark",
    "julia", "steven", "rachel", "andrew", "megan", "peter", "lucy", "paul",
    "grace", "george", "chloe", "jack", "kate", "sam", "lily", "ben", "mia",
    "tom", "zoe", "nick", "amy", "adam", "ella", "matt", "ava", "josh",
    "ruby", "luke", "isla", "max", "leah", "leo", "nina", "oscar", "eva",
    "henry", "alice", "oliver", "clara", "jacob", "sofia", "noah", "ivy",
    "liam", "rose", "ethan", "maya", "lucas", "aria", "mason", "nora",
    "somchai", "malee", "niran", "ploy", "arthit", "kanya", "pim", "nok",
]

# ---------------------------------------------------------------------------
# Keyboard rows used for keyboard-walk detection (QWERTY, unshifted & shifted).
# A "walk" is a run of physically adjacent keys along one row.
# ---------------------------------------------------------------------------
KEYBOARD_ROWS = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./",
    "~!@#$%^&*()_+",
    "QWERTYUIOP{}|",
    'ASDFGHJKL:"',
    "ZXCVBNM<>?",
]

# Column-wise "vertical" walks such as 1qaz / 2wsx (common in weak passwords).
KEYBOARD_COLUMNS = [
    "1qaz", "2wsx", "3edc", "4rfv", "5tgb", "6yhn", "7ujm", "8ik,", "9ol.",
    "0p;/",
]

# Leetspeak substitutions: symbol/digit -> letter it usually stands for.
LEET_MAP = {
    "4": "a", "@": "a", "8": "b", "(": "c", "3": "e", "6": "g", "9": "g",
    "1": "i", "!": "i", "|": "l", "0": "o", "5": "s", "$": "s", "7": "t",
    "+": "t", "2": "z", "%": "x",
}

# The reverse direction, used by the generator to "leetify" words.
LEET_FORWARD = {
    "a": ["4", "@"], "e": ["3"], "i": ["1", "!"], "o": ["0"], "s": ["5", "$"],
    "t": ["7"], "g": ["9"], "b": ["8"], "l": ["1"],
}

SYMBOLS = "!@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~"


def _dedupe(seq):
    """Return the list with duplicates removed, keeping first-occurrence order."""
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


COMMON_PASSWORDS = _dedupe(COMMON_PASSWORDS)
ENGLISH_WORDS = _dedupe(ENGLISH_WORDS)
NAMES = _dedupe(NAMES)

# Rank dictionaries (1-based) used by the guess estimator.
COMMON_RANK = {w: i + 1 for i, w in enumerate(COMMON_PASSWORDS)}
WORD_RANK = {w: i + 1 for i, w in enumerate(ENGLISH_WORDS)}
NAME_RANK = {w: i + 1 for i, w in enumerate(NAMES)}

# All attacker dictionaries in one place: name -> rank map.
DICTIONARIES = {
    "common": COMMON_RANK,
    "english": WORD_RANK,
    "names": NAME_RANK,
}
