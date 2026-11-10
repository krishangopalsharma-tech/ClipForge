"""Hinglish romanisation — Devanagari speech written in Latin letters.

Whisper transcribes Hindi/Urdu in Devanagari. Most caption presets are Latin
display faces (Anton, Bangers, Bebas…), so a Devanagari caption drops out of the
preset into whatever libass finds — at best the plain Noto Sans Devanagari in
``assets/fonts``, at worst tofu boxes.
Romanising the transcript keeps the presets: captions read "aap kaise hain",
which is how the audience types the language anyway.

It is deliberately a *readable approximation*, not academic transliteration
(the rules started in ClipForge, were refined against real Hindi speech in
OpenShorts — schwa deletion, loanwords, attached punctuation — and are back
here):

  1. Devanagari is an abugida: a bare consonant carries an inherent short
     ``a``, a matra replaces it, and virama (``्``) removes it. ``_ROMAN``
     below is the per-codepoint table, ``_romanize_word`` walks the syllables.
  2. The TRAILING inherent schwa is dropped (``ghara`` -> ``ghar``,
     ``dosta`` -> ``dost``): Hindi deletes it in speech and keeping it is the
     single thing that makes naive transliteration look wrong. Long ``ā`` is
     carried as the marker ``\x00`` until after that trim, so a genuine long
     vowel ending (``kyaa``) is never mistaken for a schwa.
  3. Long ``ī``/``ū`` fold to ``i``/``u`` (people type "ki", not "kii"), while
     ``ā`` doubles to ``aa`` ("aap", "kaam", "yaar").

Word-level, so caption timings survive: each timed token is romanised on its
own and its LEADING SPACE (the word-boundary convention the whole pipeline
relies on) is preserved.

No third-party dependency: the mapping is 60 lines of table and the rest of
indic-transliteration's machinery (IAST round-tripping, Sanskrit sandhi) buys
nothing here.
"""

import re
import unicodedata

# Any Devanagari codepoint — used to skip romanising pure-ASCII/number tokens.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Long 'ā' parked on a sentinel until the trailing-schwa trim has run.
_LONG_A = "\x00"

_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": _LONG_A, "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ॠ": "ri", "ऌ": "li", "ॡ": "li",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऍ": "e", "ऎ": "e", "ऑ": "o", "ऒ": "o",
}

# Matras (combining vowel signs) replace a consonant's inherent 'a'.
_MATRAS = {
    "ा": _LONG_A, "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "ॄ": "ri", "ॢ": "li", "ॣ": "li",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॅ": "e", "ॆ": "e", "ॉ": "o", "ॊ": "o",
}

# Consonants WITHOUT their inherent vowel; the walker adds 'a' or a matra.
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # Precomposed nukta forms (Urdu/Persian sounds).
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh",
    "फ़": "f", "य़": "y",
}

# A nukta after one of these makes a different sound; anything else keeps its
# base consonant (the nukta itself is then dropped).
_NUKTA = {
    "क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh",
    "फ": "f", "य": "y",
}

_SIGNS = {
    "ं": "n",   # anusvara      -> hain, main
    "ँ": "n",   # chandrabindu  -> hun
    "ः": "h",   # visarga
    "ऽ": "",    # avagraha
    "।": ".", "॥": ".",
}

_DIGITS = {c: str(i) for i, c in enumerate("०१२३४५६७८९")}

_VIRAMA = "्"
_NUKTA_SIGN = "़"

# The signs that carry sound and ride on the syllable before them, as opposed
# to the punctuation ones in _SIGNS (danda, avagraha).
_CODA_SIGNS = {"ं", "ँ", "ः"}

# Everything that takes part in a syllable. Anything else sitting at the EDGE
# of a token is punctuation and has to come off before the syllable walk:
# _units() would otherwise make it the last unit, and both the trailing-schwa
# deletion and the aa/a choice key off the LAST unit, so a single comma turned
# "सर," into "sara," and "क्या," into "kyaa,". Sentence-final words are a large
# share of caption text, so this was visible on almost every line.
_SYLLABIC = (
    set(_CONSONANTS) | set(_INDEPENDENT_VOWELS) | set(_MATRAS) | set(_DIGITS)
    | _CODA_SIGNS | {_VIRAMA, _NUKTA_SIGN}
)


def _nukta_free(word):
    """The word without nukta marks, for a spelling-insensitive lookup: whisper
    writes both फ़ोन and फोन, ज़िंदगी and जिंदगी. NFD first, because क़ arrives
    either precomposed (one codepoint) or as क + ़ — decomposing makes both the
    same two characters, and then the sign comes off."""
    return unicodedata.normalize("NFD", word).replace(_NUKTA_SIGN, "")


# English words as they are SPOKEN in Hindi, spelled back the English way.
#
# Rule-based romanisation can only spell what it hears, so an English loanword
# comes out phonetically — नेल पेंट as "nel pent", हेलो as "helo", स्टाइल as
# "staail". Hinglish speakers type the English spelling for these, and they are
# a large share of media speech, so a lookup in front of the phonetic walk buys
# more legibility than any further rule would.
#
# Per TOKEN, deliberately: each timed word is romanised on its own, so a
# two-word phrase like नेल पेंट is two entries and the caption keeps its
# per-word timings.
#
# A word that is ALSO a common Hindi word is left out on purpose, because a
# lookup cannot tell which was meant and the Hindi reading is usually the
# right one. Counted over one real episode's transcript: चीज़ ("thing")
# 4 occurrences, all of them "thing", none "cheese"; बस ("enough", "just")
# 3 occurrences, none of them the vehicle. Also excluded: पास (paas, "near"),
# हाय (an interjection of pain as often as a greeting). सर IS included —
# "sir" dominates in interview and comedy speech, and the cost when it means
# "head" is a mild "sir dard" rather than a wrong word.
_LOANWORDS = {
    # greetings and social
    "हेलो": "hello", "हैलो": "hello", "बाय": "bye",
    "थैंक्यू": "thank you", "थैंक्स": "thanks", "सॉरी": "sorry",
    "प्लीज़": "please", "ओके": "okay", "वेलकम": "welcome",
    "कांग्रेट्स": "congrats", "सर": "sir", "मैडम": "madam",
    # phones, computers, internet
    "मोबाइल": "mobile", "फ़ोन": "phone", "कंप्यूटर": "computer",
    "लैपटॉप": "laptop", "इंटरनेट": "internet", "वेबसाइट": "website",
    "ईमेल": "email", "मैसेज": "message", "वीडियो": "video",
    "ऑडियो": "audio", "फ़ोटो": "photo", "कैमरा": "camera",
    "चैनल": "channel", "सब्सक्राइब": "subscribe", "लाइक": "like",
    "कमेंट": "comment", "शेयर": "share", "ऐप": "app",
    "सॉफ्टवेयर": "software", "पासवर्ड": "password", "स्क्रीन": "screen",
    "बटन": "button", "लिंक": "link", "डाउनलोड": "download",
    "अपलोड": "upload", "ऑनलाइन": "online", "ऑफलाइन": "offline",
    "नेटवर्क": "network", "डेटा": "data", "फ़ाइल": "file",
    "फ़ोल्डर": "folder", "सर्च": "search", "गूगल": "google",
    "यूट्यूब": "youtube", "इंस्टाग्राम": "instagram", "फेसबुक": "facebook",
    "चार्जर": "charger", "बैटरी": "battery", "सिग्नल": "signal",
    # film and television
    "फ़िल्म": "film", "मूवी": "movie", "सिनेमा": "cinema", "शो": "show",
    "एपिसोड": "episode", "सीरियल": "serial", "स्टार": "star",
    "हीरो": "hero", "हीरोइन": "heroine", "डायरेक्टर": "director",
    "प्रोड्यूसर": "producer", "एक्टर": "actor", "एक्ट्रेस": "actress",
    "कॉमेडी": "comedy", "ड्रामा": "drama", "एक्शन": "action",
    "सीन": "scene", "शूटिंग": "shooting", "डायलॉग": "dialogue",
    "म्यूजिक": "music", "सॉन्ग": "song", "डांस": "dance",
    "ऑडिशन": "audition", "स्क्रिप्ट": "script", "रोल": "role",
    # looks and clothes
    "स्टाइल": "style", "स्टाइलिश": "stylish", "लुक": "look",
    "फैशन": "fashion", "मेकअप": "makeup", "नेल": "nail", "पेंट": "paint",
    "लिपस्टिक": "lipstick", "ड्रेस": "dress", "शर्ट": "shirt",
    "जींस": "jeans", "साइज़": "size", "कलर": "colour",
    # work and money
    "ऑफ़िस": "office", "जॉब": "job", "बिज़नेस": "business",
    "कंपनी": "company", "मीटिंग": "meeting", "प्रोजेक्ट": "project",
    "टीम": "team", "बॉस": "boss", "कस्टमर": "customer",
    "सैलरी": "salary", "बैंक": "bank", "अकाउंट": "account",
    "मार्केट": "market", "शॉप": "shop", "प्रॉब्लम": "problem",
    "सॉल्यूशन": "solution", "आइडिया": "idea", "प्लान": "plan",
    "मैनेजर": "manager", "इंटरव्यू": "interview", "ऑफ़र": "offer",
    # school
    "स्कूल": "school", "कॉलेज": "college", "क्लास": "class",
    "टीचर": "teacher", "स्टूडेंट": "student", "स्टडी": "study",
    "होमवर्क": "homework", "एग्जाम": "exam", "रिजल्ट": "result",
    "डिग्री": "degree", "कोर्स": "course", "ट्रेनिंग": "training",
    "बुक": "book", "पेन": "pen", "पेपर": "paper", "मार्क्स": "marks",
    # health
    "डॉक्टर": "doctor", "हॉस्पिटल": "hospital", "मेडिसिन": "medicine",
    "हेल्थ": "health", "फिटनेस": "fitness", "जिम": "gym",
    "एक्सरसाइज": "exercise", "डाइट": "diet", "प्रोटीन": "protein",
    "शुगर": "sugar", "टेंशन": "tension", "स्ट्रेस": "stress",
    "ऑपरेशन": "operation", "रिपोर्ट": "report",
    # travel
    "कार": "car", "ट्रेन": "train", "फ्लाइट": "flight",
    "एयरपोर्ट": "airport", "स्टेशन": "station", "टिकट": "ticket",
    "रोड": "road", "ट्रैफिक": "traffic", "बाइक": "bike",
    "पेट्रोल": "petrol", "होटल": "hotel", "रेस्टोरेंट": "restaurant",
    # home
    "रूम": "room", "किचन": "kitchen", "बाथरूम": "bathroom",
    "टीवी": "tv", "फ्रिज": "fridge", "लाइट": "light", "फैन": "fan",
    "पावर": "power", "बिल्डिंग": "building", "फ्लैट": "flat",
    # food
    "कॉफ़ी": "coffee", "पिज़्ज़ा": "pizza", "बर्गर": "burger",
    "सैंडविच": "sandwich", "चॉकलेट": "chocolate", "आइसक्रीम": "ice cream",
    "केक": "cake", "बिस्किट": "biscuit", "जूस": "juice",
    "ब्रेड": "bread", "बटर": "butter",
    "स्नैक्स": "snacks", "मेन्यू": "menu", "ऑर्डर": "order",
    # people and life
    "फैमिली": "family", "फ्रेंड": "friend", "पार्टी": "party",
    "वाइफ": "wife", "हसबैंड": "husband", "बेबी": "baby",
    "गर्लफ्रेंड": "girlfriend", "बॉयफ्रेंड": "boyfriend",
    "लव": "love", "लाइफ": "life", "फ्यूचर": "future",
    "ड्रीम": "dream", "लक": "luck", "चांस": "chance", "रिस्क": "risk",
    "मिस्टेक": "mistake", "सक्सेस": "success", "पब्लिक": "public",
    "पुलिस": "police", "गवर्नमेंट": "government", "कोर्ट": "court",
    # counting and comparing
    "नंबर": "number", "लेवल": "level", "टाइप": "type",
    "फर्स्ट": "first", "लास्ट": "last", "बेस्ट": "best",
    "टॉप": "top", "परसेंट": "percent", "डबल": "double",
    # time
    "टाइम": "time", "डेट": "date", "मिनट": "minute",
    "सेकंड": "second", "वीक": "week", "मंथ": "month",
    "वीकेंड": "weekend", "बर्थडे": "birthday",
    # measured as misses on a real code-switched episode
    "हेयरस्टाइल": "hairstyle", "सेक्सी": "sexy", "सेंचरी": "century",
}

# Both spellings resolve to the same word: keys are written with their nukta,
# and the fold adds the bare variant whisper also produces.
_LOANWORDS.update({
    _nukta_free(word): english for word, english in list(_LOANWORDS.items())
    if _nukta_free(word) not in _LOANWORDS
})


# Devanagari has no settled spelling for an English vowel, and whisper picks a
# different matra from one line to the next: मैसेज and मेसेज are the same
# "message", फोन and फॉन the same "phone". Measured on a code-switched episode,
# मैसेज was in the table and मेसेज was not, so the same word romanised as
# "message" on one caption and "mesej" on another. These two pairs are the ones
# whisper actually alternates; ि/ी and ु/ू are not, so they are left alone.
_MATRA_VARIANTS = (("े", "ै"),      # े / ै   (e / ai)
                   ("ो", "ॉ"))      # ो / ॉ   (o / aw)

# Generating variants blindly manufactures wrong entries, because some variant
# spellings are words in their own right: रॉड is "rod" while "road" is रोड,
# रॉल a roll rather than a role, फेन is Hindi for foam, टोप a cap. The fold
# skips those — the same discipline as the ambiguous words kept out of the
# table above. पैंट is skipped for a weaker reason: it is how "pants" is
# usually written, while the table's पेंट is "paint". The phonetic walk still
# spells it "paint" (ै IS "ai"), which no lookup can fix, but the table at
# least does not assert it.
_NO_VARIANT = {"पैंट", "रॉड", "फेन", "टोप", "रॉल"}

_LOANWORDS.update({
    variant: english
    for word, english in list(_LOANWORDS.items())
    for a, b in _MATRA_VARIANTS
    for src, dst in ((a, b), (b, a))
    if src in word
    for variant in [word.replace(src, dst)]
    if variant not in _LOANWORDS and variant not in _NO_VARIANT
})


def _split_affixes(word):
    """Split a token into (leading punctuation, syllabic core, trailing)."""
    start, end = 0, len(word)
    while start < end and word[start] not in _SYLLABIC:
        start += 1
    while end > start and word[end - 1] not in _SYLLABIC:
        end -= 1
    return word[:start], word[start:end], word[end:]


def _romanize_affix(affix):
    """Punctuation passes through, but Devanagari punctuation is still mapped
    (danda -> full stop) — peeling it must not stop it being romanised."""
    return "".join(_SIGNS.get(c, c) for c in affix)


def _units(word):
    """Split a Devanagari token into syllable units.

    Each unit is ``{"cons", "vowel", "coda", "inherent"}``: the consonant
    cluster, its vowel, any nasal/visarga sign riding on it, and whether that
    vowel is the inherent schwa (the only one schwa deletion may remove).
    Non-syllabic characters (digits, punctuation, stray Latin) become units of
    their own so they survive untouched.
    """
    units = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        base = _CONSONANTS.get(ch)
        if base is not None:
            i += 1
            # क + ़  (decomposed nukta) is the same sound as the precomposed क़.
            if i < n and word[i] == _NUKTA_SIGN:
                base = _NUKTA.get(ch, base)
                i += 1
            unit = {"cons": base, "vowel": "a", "coda": "", "inherent": True}
            if i < n and word[i] == _VIRAMA:
                unit["vowel"], unit["inherent"] = "", False
                i += 1
            elif i < n and word[i] in _MATRAS:
                unit["vowel"], unit["inherent"] = _MATRAS[word[i]], False
                i += 1
            units.append(unit)
            continue
        i += 1
        if ch in _INDEPENDENT_VOWELS:
            units.append({"cons": "", "vowel": _INDEPENDENT_VOWELS[ch],
                          "coda": "", "inherent": False})
        elif ch in _MATRAS:  # orphan matra (malformed input)
            units.append({"cons": "", "vowel": _MATRAS[ch],
                          "coda": "", "inherent": False})
        elif ch in _SIGNS:
            if units and not units[-1]["coda"]:
                units[-1]["coda"] = _SIGNS[ch]
            else:
                units.append({"cons": _SIGNS[ch], "vowel": "",
                              "coda": "", "inherent": False})
        elif ch in _DIGITS:
            units.append({"cons": _DIGITS[ch], "vowel": "",
                          "coda": "", "inherent": False})
        elif ch not in (_VIRAMA, _NUKTA_SIGN):
            units.append({"cons": ch, "vowel": "", "coda": "",
                          "inherent": False})
    return units


def _delete_schwas(units):
    """Apply Hindi schwa deletion in place — ghar, not ghara; ladki, not ladaki.

    Two rules, in this order (the order is the algorithm: the final deletion
    is what stops the second rule from eating the schwa in samajh):

      1. The word-final inherent schwa always goes, in any word of more than
         one syllable. This is the single biggest source of "transliteration
         looks wrong": ghara, dosta, kapila.
      2. Then the standard VC_CV rule — an inherent schwa surrounded by a
         vowel before and a real vowel after is dropped (ladaki -> ladki,
         matalaba -> matlab), left to right so a schwa already deleted no
         longer supplies the vowel its neighbour would need. A CLOSED syllable
         before it blocks the rule: zindagi keeps its schwa, zindgi is not a
         word anyone types.
    """
    if len(units) > 1 and units[-1]["inherent"] and not units[-1]["coda"]:
        units[-1]["vowel"], units[-1]["inherent"] = "", False
    for i in range(1, len(units) - 1):
        if not units[i]["inherent"] or units[i]["coda"]:
            continue
        if (units[i - 1]["vowel"] and not units[i - 1]["coda"]
                and units[i + 1]["vowel"]):
            units[i]["vowel"], units[i]["inherent"] = "", False


def _romanize_word(word):
    """Romanise a single Devanagari token to readable Hinglish."""
    if not _DEVANAGARI.search(word):
        return word  # numbers / Latin / punctuation pass straight through

    head, core, tail = _split_affixes(word)
    if not core:
        return _romanize_affix(word)

    # An English word gets its English spelling back rather than a phonetic
    # guess at it ("nail paint", not "nel pent").
    english = _LOANWORDS.get(core) or _LOANWORDS.get(_nukta_free(core))
    if english:
        return _romanize_affix(head) + english + _romanize_affix(tail)

    units = _units(core)
    _delete_schwas(units)

    out = []
    for idx, unit in enumerate(units):
        vowel = unit["vowel"]
        # Long 'aa' doubles inside a word (aap, kaam, pyaar) but not at the
        # end, where nobody types the second one: sharma, kya, achcha, hua —
        # and that holds for a ONE-syllable word too (ka, tha, na, ja, ya,
        # among the most frequent words in the language), which is why a
        # single unit still shortens as long as a consonant carries the vowel.
        # Two things keep both letters: a bare vowel that IS the whole word
        # (आ = "aa") and a nasalised ending, where the long vowel is audible
        # and typed (हाँ = "haan", माँ = "maan").
        if vowel == _LONG_A:
            last = idx == len(units) - 1
            final_a = (last and not unit["coda"]
                       and (len(units) > 1 or unit["cons"]))
            vowel = "a" if final_a else "aa"
        out.append(unit["cons"] + vowel + unit["coda"])

    roman = "".join(out).replace(_LONG_A, "aa")
    # च्छ is ch + chh; nobody writes the doubled h twice (achcha, not achchha).
    roman = roman.replace("chchh", "chch")

    # Fold any stray diacritic that came in with the source text.
    roman = "".join(
        c for c in unicodedata.normalize("NFKD", roman)
        if not unicodedata.combining(c)
    ).lower()
    return _romanize_affix(head) + roman + _romanize_affix(tail)


def to_roman(text):
    """Romanise a run of text token-by-token (whitespace preserved)."""
    if not text or not _DEVANAGARI.search(text):
        return text
    # Split on whitespace but KEEP the separators: a word token's leading
    # space is the pipeline's word-boundary marker.
    parts = re.split(r"(\s+)", text)
    return "".join(p if p.isspace() else _romanize_word(p) for p in parts)


def romanize_transcript(transcript):
    """Return a copy of a transcript with every text field romanised.

    Handles both transcript shapes in this repo: segments carrying their own
    ``words`` (transcribe_backends) and a flat top-level ``words`` list. Word
    timings are untouched, so captions stay in sync. ``language`` becomes
    ``"hinglish"`` so downstream prompts and metadata say what the text is.
    """
    if not transcript:
        return transcript

    out = dict(transcript)
    out["language"] = "hinglish"
    if transcript.get("text"):
        out["text"] = to_roman(transcript["text"])
    if isinstance(transcript.get("words"), list):
        out["words"] = [
            {**w, "word": to_roman(w.get("word", ""))} for w in transcript["words"]
        ]
    out["segments"] = [
        {
            **s,
            "text": to_roman(s.get("text", "")),
            **({"words": [{**w, "word": to_roman(w.get("word", ""))}
                          for w in s["words"]]}
               if isinstance(s.get("words"), list) else {}),
        }
        for s in (transcript.get("segments") or [])
    ]
    return out


def available():
    """Always True: there is no backend to import any more (kept for callers
    written when this module wrapped the indic-transliteration package)."""
    return True


def romanize_if_needed(transcript):
    """Romanise a transcript only when it actually carries Devanagari.

    For the reuse paths (a Thumbnail Studio handover, the checkpoint an
    interrupted run left behind), where the text may already be Latin — an
    English transcript must not be retagged as Hinglish.
    """
    text = (transcript or {}).get("text") or ""
    if not text:
        text = " ".join(str(s.get("text") or "")
                        for s in (transcript.get("segments") or []))
    if not _DEVANAGARI.search(text):
        return transcript
    return romanize_transcript(transcript)
