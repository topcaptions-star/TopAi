# Alvar Caption API

שרת Backend קטן ל-Render. הוא מחזיק את `SPEECHMATICS_API_KEY` ומעולם לא שולח אותו לפלאגין.

## Render

1. צור Web Service חדש וחבר את התיקייה הזו ל-Repository.
2. Build Command: `npm install`
3. Start Command: `npm start`
4. הוסף ב-Environment:

```text
SPEECHMATICS_API_KEY=המפתח שלך
ALLOWED_ORIGIN=*
MAX_UPLOAD_MB=250
```

אל תכניס את המפתח ל-GitHub, ל-ZXP או לקוד JavaScript.

## Endpoints

- `GET /health` — בדיקת חיבור.
- `POST /transcribe` — multipart field בשם `media`; שדות אופציונליים: `language` (`auto`, `he`, `en`, וכו'), `model` (`melia-1`, `standard`, `enhanced`) ו-`maxChars`.

השרת מוחק את קובץ המדיה הזמני לאחר סיום הבקשה ומחזיר JSON עם `words` ו-`segments`.

## בדיקה

```bash
curl https://YOUR-SERVICE.onrender.com/health
curl -X POST https://YOUR-SERVICE.onrender.com/transcribe \
  -F media=@sample.wav -F language=he -F model=melia-1
```
