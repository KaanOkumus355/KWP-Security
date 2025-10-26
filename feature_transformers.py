# -------------------------------------------------
# feature_transformers.py
# -------------------------------------------------
import re
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------------------------------------------
# 1️⃣ URL flag – 1 if a URL is present, else 0
# -----------------------------------------------------------------
class UrlFlagTransformer(BaseEstimator, TransformerMixin):
    """1‑column sparse matrix (n_samples × 1) – 1 if a URL is present, else 0."""
    def __init__(self):
        self.regex = re.compile(r"https?://\S+|www\.\S+")
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        flags = [1 if self.regex.search(txt) else 0 for txt in X]
        # reshape to a column vector (n,1)
        return sparse.csr_matrix(np.array(flags).reshape(-1, 1))


# -----------------------------------------------------------------
# 2️⃣ Keyword counter – counts of the long keyword list you supplied
# -----------------------------------------------------------------
class KeywordFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Counts of a short, fixed list of phishing‑related keywords.
    The list can be overridden when the transformer is instantiated.
    """
    def __init__(self, keywords=None):
        if keywords is None:
            keywords = [
                # ----- Security / Verification -----
                "secure", "security", "verification", "authenticate", "credentials",
                "login", "sign in", "sign-in", "password", "passcode", "account locked",
                "reset", "confirm identity", "2fa", "otp", "token", "unauthorized",
                "suspended", "compromised", "recovery", "reactivate", "unlock",

                # ----- Urgency / Threat -----
                "urgent", "immediate action", "act now", "attention", "important",
                "final notice", "warning", "alert", "deadline", "expire", "expiration",
                "failure to", "as soon as possible", "limited time", "critical",
                "respond now", "take action", "avoid suspension",

                # ----- Financial / Payment -----
                "payment", "bank", "credit", "debit", "card", "account", "balance",
                "transfer", "funds", "invoice", "transaction", "refund", "prize",
                "lottery", "bonus", "cash", "payment declined", "billing", "charges",
                "amount", "update payment", "paypal", "crypto", "btc", "ethereum",

                # ----- Links / Domains -----
                "click", "click here", "link", "login page", "portal", "secure link",
                "http", "https", "tinyurl", "bit.ly", "goo.gl", "shorturl", "redirect",
                "verify link", "access link", "webpage", "site", "domain", "url",

                # ----- Impersonation / Authority -----
                "support", "helpdesk", "service", "customer care", "administrator",
                "admin", "it department", "security team", "official", "government",
                "irs", "amazon", "microsoft", "apple", "google", "facebook",
                "bank of america", "paypal", "ebay", "netflix", "delivery", "fedex",
                "dhl", "post office",

                # ----- Offers / Rewards -----
                "congratulations", "you won", "winner", "free", "offer", "gift",
                "reward", "claim now", "exclusive", "deal", "promotion", "discount",
                "voucher", "redeem", "bonus", "lucky", "selected", "special offer",

                # ----- Information Requests -----
                "provide", "confirm", "update information", "submit", "enter", "details",
                "identity", "ssn", "social security", "birth date", "personal data",
                "confidential", "private", "information request", "upload document",

                # ----- Psychological / Manipulation -----
                "trust", "guarantee", "official notice", "avoid penalty", "you must",
                "we regret", "we noticed", "we detected", "your cooperation",
                "due to unusual activity", "for your safety", "to protect you",

                # ----- Money symbols / generic indicators -----
                "$", "€", "£"
            ]
        self.keywords = [kw.lower() for kw in keywords]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for txt in X:
            txt_lc = txt.lower()
            rows.append([txt_lc.count(kw) for kw in self.keywords])
        return sparse.csr_matrix(rows)


# -----------------------------------------------------------------
# 3️⃣ Domain whitelist – 1 if the URL domain is in a trusted list
# -----------------------------------------------------------------
class DomainWhitelistTransformer(BaseEstimator, TransformerMixin):
    """
    Returns a single binary column:
        1 → the URL domain is in the *whitelist* (trusted site)
        0 → otherwise (or no URL at all)
    """
    def __init__(self, whitelist=None):
        if whitelist is None:
            whitelist = [
                "amazon.com", "paypal.com", "google.com", "facebook.com",
                "apple.com", "microsoft.com", "ebay.com", "netflix.com",
                "github.com", "stackoverflow.com", "nytimes.com",
                "stripe.com", "adobe.com", "linkedin.com"
            ]
        self.whitelist = set(w.lower() for w in whitelist)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X – iterable of raw strings.
        Extract the first URL (if any), get its domain, and check the whitelist.
        """
        pattern = re.compile(r"https?://([^/]+)")
        flags = []
        for txt in X:
            m = pattern.search(txt)
            if m:
                domain = m.group(1).lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                flags.append(1 if domain in self.whitelist else 0)
            else:
                flags.append(0)          # no URL → not whitelisted
        return sparse.csr_matrix(np.array(flags).reshape(-1, 1))


# -----------------------------------------------------------------
# 4️⃣ First‑person pronoun flag – 1 if the text contains I / my / we …
# -----------------------------------------------------------------
class FirstPersonPronounTransformer(BaseEstimator, TransformerMixin):
    """1 if a first‑person pronoun appears, else 0."""
    def __init__(self):
        self.pronouns = {"i", "me", "my", "mine", "we", "our", "ours", "bro"}
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        flags = []
        for txt in X:
            tokens = set(txt.lower().split())
            flags.append(1 if tokens & self.pronouns else 0)
        return sparse.csr_matrix(np.array(flags).reshape(-1, 1))
