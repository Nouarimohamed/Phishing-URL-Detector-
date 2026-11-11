"""URL -> features. Shared by train.py and predict.py so training and the app always agree.

Only the DOMAIN is analysed. The scheme (http/https), a leading "www." and the path are
ignored on purpose: in the PhiUSIIL dataset EVERY legitimate URL is "https://www.<domain>"
with no path, so using those parts teaches the model a shortcut ("no www / has a path =
phishing") instead of real phishing signals.
"""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd


def split_domain(url: str):
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    netloc = urlparse(url).netloc.lower()
    has_at = "@" in netloc
    domain = netloc.split("@")[-1].split(":")[0]
    domain = re.sub(r"^www\.", "", domain)
    return domain, has_at


def _entropy(s: str) -> float:
    n = len(s) or 1
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def tld_of(url: str) -> str:
    labels = split_domain(url)[0].split(".")
    return labels[-1] if len(labels) > 1 else ""


def features_dict(url: str, tlds: list) -> dict:
    domain, has_at = split_domain(url)
    try:
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    labels = domain.split(".")
    tld = labels[-1] if len(labels) > 1 else ""
    sld = labels[-2] if len(labels) > 1 else domain  # e.g. "google" in google.com
    n = len(domain) or 1
    digits = sum(c.isdigit() for c in domain)

    f = {
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "TLDLength": len(tld),
        "NoOfSubDomain": max(len(labels) - 2, 0),
        "NoOfDigitsInDomain": digits,
        "DigitRatioInDomain": digits / n,
        "NoOfHyphensInDomain": domain.count("-"),
        "NoOfDotsInDomain": domain.count("."),
        "HasAtSymbol": int(has_at),
        "LongestLabel": max(map(len, labels)),
        "SLDLength": len(sld),
        "SLDEntropy": _entropy(sld),
        "SLDVowelRatio": sum(c in "aeiou" for c in sld) / (len(sld) or 1),
    }
    for t in tlds:  # one column per frequent TLD (.com, .xyz, .tk ...)
        f["tld_" + t] = int(tld == t)
    return f


def extract_features(url: str, tlds: list, columns: list) -> pd.DataFrame:
    return pd.DataFrame([features_dict(url, tlds)])[columns]
