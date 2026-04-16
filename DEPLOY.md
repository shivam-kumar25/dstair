# DSTAIR — Deployment Guide (GoDaddy)

**What you need before starting:**

- A GoDaddy account with a **Linux Web Hosting** plan (Economy, Deluxe, or Ultimate)
- Your GitHub repo URL (the one containing this project)

---

## Step 2 — Create a Python App on GoDaddy

1. Log into [godaddy.com](https://godaddy.com) → **My Products** → next to your hosting plan click **Manage**
2. This opens **cPanel**. Scroll down to the **Software** section and click **Setup Python App**
3. Click **Create Application** and fill in exactly:

   | Field | What to enter |
   | --- | --- |
   | Python version | `3.11` |
   | Application root | `dstair` |
   | Application URL | `/` |
   | Application startup file | `passenger_wsgi.py` |
   | Application Entry point | `application` |

4. Click **Create**

GoDaddy will create a folder called `dstair` on your server. You'll replace it with your code in the next step.

---

## Step 3 — Open the GoDaddy Terminal

Still inside cPanel, scroll down and click **Terminal** (under the Advanced section).

A black terminal window will open — this is how you control your server.

---

## Step 4 — Clone Your GitHub Repo

In the terminal, run these commands **one at a time**, pressing Enter after each:

```bash
rm -rf ~/dstair
```

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git dstair
```

> Replace `YOUR_GITHUB_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub details.
> For example: `git clone https://github.com/johnsmith/dstair-project.git dstair`

---

## Step 5 — Install the App's Packages

Still in the terminal, run these three commands one at a time:

```bash
source ~/virtualenv/dstair/3.11/bin/activate
```

```bash
cd ~/dstair
```

```bash
pip install -r requirements.txt
```

Wait for it to finish — it may take a minute or two.

---

## Step 6 — Create Your Secret Settings File

First, generate a secret key by running:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the long string of letters and numbers it prints — that is your secret key.

Now create the settings file:

```bash
nano ~/dstair/.env
```

A text editor opens. Type in the following, replacing the placeholder values:

```
SECRET_KEY=paste_your_secret_key_here
FLASK_ENV=production
DATABASE_URI=sqlite:////home/YOUR_CPANEL_USERNAME/dstair/instance/database.db
GROQ_API_KEY=paste_your_groq_key_here
```

> **How to find your cPanel username:** look at the top-right corner of cPanel — it shows your username there.
>
> **DATABASE_URI uses 4 slashes** — that is not a typo.

To save: press `Ctrl + X`, then `Y`, then `Enter`.

Then lock the file so only your app can read it:

```bash
chmod 600 ~/dstair/.env
```

---

## Step 7 — Set File Permissions

Run these two commands:

```bash
chmod 755 ~/dstair/instance
chmod 644 ~/dstair/instance/database.db
```

This allows the app to read and write to the database.

---

## Step 8 — Restart the App

Go back to **cPanel → Setup Python App**, click on your `dstair` app, and click **Restart**.

Then open your domain in a browser — the site should be live.

---

## Something Went Wrong?

Check the error log: **cPanel → Logs → Error Log**

| What you see | What to do |
| --- | --- |
| White page or 500 error | Check the error log for the exact message |
| `FATAL: SECRET_KEY insecure` | Your `.env` file is missing or `SECRET_KEY` is not filled in |
| `unable to open database` | Check `DATABASE_URI` in `.env` — 4 slashes, and your exact cPanel username |
| `ModuleNotFoundError` | Run Step 5 again |

---

## Updating the Site Later

Whenever you make changes and push to GitHub, update GoDaddy by opening the terminal and running:

```bash
cd ~/dstair && git pull origin main && touch passenger_wsgi.py
```

That's it — the site will reload with your latest changes.
