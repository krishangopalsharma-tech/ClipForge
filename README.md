# ClipForge

Kisi bhi video ko **captioned vertical shorts** bana do — **100% aapke PC par**,
koi API key nahi. NVIDIA card hua to khud tez (GPU) chalega, warna CPU par.

## Chalane ka tareeqa (Windows)

1. **[ClipForge.bat download karein](https://raw.githubusercontent.com/Ai-Haris/clipforge/main/packaging/ClipForge.bat)**
   *(link khulne par: Ctrl+S se save karein — ek khaali folder mein rakhein.)*
2. **`ClipForge.bat` par double-click** karein.
3. Bas — pehli baar sab kuch (Python, libraries, ffmpeg, model) khud download hota
   hai (~5–10 min, internet chahiye). Phir browser khud khul jata hai. Agli dafa turant.

> Kuch alag install karne ki zarurat **nahi** — Python bhi nahi.

## Zaruri baatein

- **Windows 10/11 (64-bit).**
- Pehli baar **internet** chahiye (sab kuch download hota hai). Baad mein apni file
  upload karke offline bhi chala sakte hain.
- **GPU:** NVIDIA card hua to app "Compute → Auto" par khud GPU use karti hai (bahut
  tez). Card na ho to CPU (thodi slow) — kaam phir bhi hota hai.

## Masla aaye to

- **SmartScreen / Antivirus** roke → *More info → Run anyway* (bat unsigned hai).
- **`VCRUNTIME140.dll` missing** → Microsoft Visual C++ Redistributable (x64) install
  karein, phir dobara `ClipForge.bat`.
- **Port 8000 busy** → koi aur ClipForge pehle se chal rahi hogi; usko band karein.
- Pehli run par **"Connecting…"** kaafi der dikhe → model (~1.5 GB) download ho raha
  hai, sabar karein. Ek dafa ka kaam hai.

## Studio mode — moments Claude ya Gemini se chunwayein

Default selector ek local heuristic hai (word density, sawal, strong words). Theek
hai, lekin woh nahi samajhta ke koi moment *viral* kyun hoga. Studio mode woh
faisla aap ke apne AI chat par chhod deta hai — koi API key nahi, koi extra model
download nahi, GPU sirf Whisper ke liye.

**`http://127.0.0.1:8000/studio`** kholein:

1. **Video daalein** — local Whisper transcribe karta hai. File apne *content* se
   pehchani jaati hai, is liye wohi video dobara kabhi transcribe nahi hoti.
2. **Transcript download karein** — numbered, timestamped lines. Word-level
   timings hata di jati hain (captions ke liye chahiye, moment choose karne ke
   liye nahi): 21-minute video = ~19 KB, chat mein aaram se paste ho jata hai.
3. **Prompt copy karein** aur transcript ke saath Claude / Gemini ko dein.
4. **Jo JSON aaye woh wapas paste karein** — prose ya ``` fence ho to bhi chalega.
   Check dabayein: har clip ka time, length aur alfaz render se *pehle* dikh jate hain.
5. **Render** — wohi ffmpeg cut + reframe + captions.

Model **segment ids** deta hai, timestamps nahi (`{"start_seg": 142, "end_seg": 158}`).
Ids ClipForge apne transcript se resolve karta hai, is liye time bilkul theek rehta
hai — model number galat likh de to bhi clip apne hook par hi shuru hoti hai.

API: `GET /api/transcript/{source_id}` · `GET /api/clip-prompt` ·
`POST /api/manual-windows` · `POST /api/generate` mein `manual_windows`.

## Yeh kya hai

Local pipeline: **yt-dlp** (video laata hai) → **faster-whisper** (word-level
captions, local) → local heuristic (ya aap ke chune hue moments) → **ffmpeg** se
cut + reframe + captions burn. Video aur transcript aap ke PC se bahar nahi jate;
Studio mode mein transcript aap khud apne chat mein paste karte hain.

Banaya: **The Haris** — [Instagram](https://www.instagram.com/theharis.ai/) · [YouTube](https://www.youtube.com/@harisailab)
