#!/usr/bin/env python3
"""
Generate expanded ground truth with dual labels (relevance + instructional quality).

Labeled pool: 86 representative videos across all categories.
Queries: 15 queries spanning cooking, fitness, crafts, tech, tiffany, history,
         photography, coding, drawing, skincare, music production.

Grades:
  relevance:     0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant
  instructional: 0=not instructional, 1=minimal, 2=moderate, 3=highly instructional (tutorial/step-by-step)
"""

import json
import os

# ── Labeled video pool (86 videos) ──────────────────────────────────────────
# Original 34 + 52 new videos covering all categories

LABELED_VIDEOS = [
    # --- Original 34 ---
    # Cooking
    "yt_6Db60rwdOuY",  # How To Cook Blue Lobster
    "yt_j19HFPplMQ0",  # Knife Skills Progression
    "yt_f-twQeqQAZE",  # How To Cook Steak Perfectly
    "yt_AmC9SmCBUj4",  # Gordon Ramsay's ULTIMATE Steak
    "yt_dv7BDkQmvhA",  # How to Cook with Ice
    "yt_yzDAApWOgYg",  # Cooking basics for Newbies Part 10
    "yt_VAHnSiSBf5g",  # If your dad never taught you how to cook
    "yt_WdYd_UCiZsM",  # How to Cook the Perfect Rice
    "yt_xxzNu82YL10",  # How To Cook Using a Dishwasher
    # Fitness
    "yt_jx9I-1D6GLs",  # 5 Minutes Quick Workout for Beginners
    "yt_LFF7iCW5Y2E",  # Full Body Workout WITHOUT EQUIPMENT
    "yt_-KKfHX56OTY",  # 3 EXERCISES FOR A TONED CORE
    "yt_Qn1voplJI4I",  # Quick and Simple Workout for Beginners
    "yt_B1KbolzpWD4",  # The Daily 50 Workout For Beginners
    # Crafts
    "yt_OEWtQLQ9czs",  # Cardboard SURFboards DIY
    "yt_hB-AAGcMsO4",  # DIY Cardboard RAGDOLL
    "yt_Y0t4mUr5vFI",  # Paper Star tutorial
    "yt_fcRPECfjHjg",  # DIY Moon Lamp
    "yt_TCM_eADif8A",  # hanging paper doll tutorial
    "yt_IkHYli39Exg",  # Make your own underwater world
    # Tech
    "yt_DVJDUAgPKR4",  # A PC Tip I Guarantee You Didn't Know
    "yt_9csDczj86v8",  # CLEAN YOUR GPU (in 10 steps)
    "yt_zhgGaIOJwBQ",  # This HDMI cable has an RTX 3090 chip?
    "yt_FPe3TAKOT9Y",  # This Common Mistake Can Ruin Your Monitor
    "yt_tj-tp_XDCqE",  # Why you NEED to install RAM like this
    "yt_Y2ZG9SrxNo8",  # How to Upgrade Your Graphics Card
    "yt_3LiFEaPUiNQ",  # Android Manufacturers HATE this SECRET App
    "yt_ka4nR0wqpuY",  # Coolest Sleeper PC at Whale LAN
    "yt_jHw5FsWYgBY",  # Are you maximizing your RAM speed?
    # History
    "yt_Xv-_c3WtY8M",  # Crazy Facts About Queen Cleopatra
    "yt_ibp1HgWoBPA",  # Kennedy's Crappy Parenting
    # Tiffany
    "yt_sW5QGWCQgHY",  # $16B Tiffany & Co. Origin Story
    "yt_tft9uUJFrkE",  # How Tiffany & Co was Created?!
    "yt_VmRvzOniIdg",  # Tiffany & Co Is Lying To You

    # --- New 52 videos ---
    # Cooking (3 more)
    "yt_-0Zp_fL3jCo",  # How To Cook Perfect Scallops
    "yt_6ezSIJZt6io",  # How To Cook With Stainless Steel
    "yt_ayV17R7gE8M",  # How to Cook Tender, Juicy Chicken
    # Fitness (3 more)
    "yt_wCPvebDq2yw",  # quick full body kettlebell workout
    "yt_xLpbXcIwQpk",  # Full Body Workout at Home
    "yt__6WERjVqBwE",  # THIS type of AT HOME beginner workouts
    # Crafts (3 more)
    "yt_UqiLaQXY5tY",  # WHISTLE ORIGAMI EASY CRAFT TUTORIAL
    "yt_duf4Ra5-0pE",  # ORIGAMI CROSSBOW PAPER CRAFT TUTORIAL
    "yt_pgxmPoVyj80",  # Useless Paper Boat Easy DIY Craft
    # History (4 more)
    "yt_A9ejBOhq-I8",  # How different countries Teach History
    "yt_aZw_u60v_5U",  # Why didn't Alexander conquer Rome?
    "yt_rcPVDigl1_c",  # In awe of Reggianini's work!! #history #art
    "yt_XE_NjAF7Le0",  # A Short History of Norway
    # Tiffany (6 more)
    "yt_xX2ejPGswBM",  # Paloma Picasso on Her Personal Story
    "yt_6HwBIB5S55U",  # The History of Tiffany & Co
    "yt_HMc3ZGNmF3o",  # Why Tiffany Jewelry Is So Expensive
    "yt_7BPU0F4da5w",  # the history of the tiffany toggle necklace
    "yt_EBEr2g2-DSk",  # The Hidden History of Tiffany & Co.
    "yt_aGydCvIJKNw",  # WHY THE TIFFANY OPEN HEART PENDANT
    # Photography (5)
    "yt_souKz302luU",  # Unlocking the Secret to Sharpness
    "yt_L7N8U7emCVU",  # 5 Tips to INSTANTLY Take Better Photos
    "yt_l2o-GgFyLBg",  # EASY Portrait Photography Tip
    "yt__Xk3XIyYsO8",  # Teaching you photography tips
    "yt_qnRsO7uTFRc",  # Try this mobile photography trick
    # Music production (4)
    "yt_QqXWu5O3uyI",  # How to Make Really Good Melodies
    "yt_hSFEv6kbC0c",  # You only need 3 things to start producing
    "yt_BG50BcFctyc",  # HOW TO MAKE A SONG IN BANDLAB
    "yt_JKd8h7fJMvQ",  # Write good melodies with rhythm
    # Coding (5)
    "yt_NGlX-1jhYJE",  # How to Learn to Code
    "yt_u5vWFfZ6Csc",  # html tutorial for beginners
    "yt_BVyxC68kHdc",  # How to write a for loop in python
    "yt_cTP6Bm4MXZU",  # The best way to learn coding
    "yt_3fkP88FDgaI",  # coding is shockingly uncomplicated
    # Drawing (3)
    "yt_y6KjkIbz9OY",  # How to draw lips
    "yt_1OLI4qiHh9Q",  # How to draw a nose
    "yt_P39ksl_9EFc",  # how I draw faces / beginner friendly
    # Skincare (3)
    "yt_P2VDMq1mF24",  # How to Layer your Nighttime Skincare
    "yt_CYizyqEfk1s",  # CORRECT ORDER TO APPLY SKINCARE
    "yt_SKOOSv-9pYQ",  # how to: nighttime skincare routine
    # Yoga (2)
    "yt_M-s835Zq1nM",  # How to do a yoga backbend
    "yt_XyaMyzLkOlA",  # Before you start yoga you need to know
    # Language (2)
    "yt_gybAlzzE_lU",  # How to Learn a Language Faster
    "yt_E1xsO_hd43s",  # Learn Japanese At Home
    # Makeup (2)
    "yt_YuQW_HSXcBI",  # Tutorial SFX makeup
    "yt_raEnFm0sdnc",  # BASIC MAKEUP TUTORIAL FOR BEGINNERS
    # Finance (2)
    "yt_37F9l3Q1HzE",  # Do These 5 Things To Win With Money
    "yt_EyBcps1e38o",  # 3 Easy Tips To Become Financially Independent
    # Guitar (2)
    "yt_9ZXP2mIgK_o",  # How to Practice Scales
    "yt_q4wtHuEj-eA",  # 3 TIPS For Your Barre Chords
    # Gardening (1)
    "yt_6lswAFsS6qs",  # 5 Tips for Growing Curry Leaf Plant
    # Math (1)
    "yt_Wkds7WSy0po",  # China-USA Multiplication Tricks
]

# ── Category assignments ────────────────────────────────────────────────────
CATEGORIES = {
    "cooking": [
        "yt_6Db60rwdOuY", "yt_j19HFPplMQ0", "yt_f-twQeqQAZE", "yt_AmC9SmCBUj4",
        "yt_dv7BDkQmvhA", "yt_yzDAApWOgYg", "yt_VAHnSiSBf5g", "yt_WdYd_UCiZsM",
        "yt_xxzNu82YL10", "yt_-0Zp_fL3jCo", "yt_6ezSIJZt6io", "yt_ayV17R7gE8M",
    ],
    "fitness": [
        "yt_jx9I-1D6GLs", "yt_LFF7iCW5Y2E", "yt_-KKfHX56OTY", "yt_Qn1voplJI4I",
        "yt_B1KbolzpWD4", "yt_wCPvebDq2yw", "yt_xLpbXcIwQpk", "yt__6WERjVqBwE",
    ],
    "crafts": [
        "yt_OEWtQLQ9czs", "yt_hB-AAGcMsO4", "yt_Y0t4mUr5vFI", "yt_fcRPECfjHjg",
        "yt_TCM_eADif8A", "yt_IkHYli39Exg", "yt_UqiLaQXY5tY", "yt_duf4Ra5-0pE",
        "yt_pgxmPoVyj80",
    ],
    "tech": [
        "yt_DVJDUAgPKR4", "yt_9csDczj86v8", "yt_zhgGaIOJwBQ", "yt_FPe3TAKOT9Y",
        "yt_tj-tp_XDCqE", "yt_Y2ZG9SrxNo8", "yt_3LiFEaPUiNQ", "yt_ka4nR0wqpuY",
        "yt_jHw5FsWYgBY",
    ],
    "history": [
        "yt_Xv-_c3WtY8M", "yt_ibp1HgWoBPA", "yt_A9ejBOhq-I8", "yt_aZw_u60v_5U",
        "yt_rcPVDigl1_c", "yt_XE_NjAF7Le0",
    ],
    "tiffany": [
        "yt_sW5QGWCQgHY", "yt_tft9uUJFrkE", "yt_VmRvzOniIdg", "yt_xX2ejPGswBM",
        "yt_6HwBIB5S55U", "yt_HMc3ZGNmF3o", "yt_7BPU0F4da5w", "yt_EBEr2g2-DSk",
        "yt_aGydCvIJKNw",
    ],
    "photography": [
        "yt_souKz302luU", "yt_L7N8U7emCVU", "yt_l2o-GgFyLBg", "yt__Xk3XIyYsO8",
        "yt_qnRsO7uTFRc",
    ],
    "music": [
        "yt_QqXWu5O3uyI", "yt_hSFEv6kbC0c", "yt_BG50BcFctyc", "yt_JKd8h7fJMvQ",
    ],
    "coding": [
        "yt_NGlX-1jhYJE", "yt_u5vWFfZ6Csc", "yt_BVyxC68kHdc", "yt_cTP6Bm4MXZU",
        "yt_3fkP88FDgaI",
    ],
    "drawing": ["yt_y6KjkIbz9OY", "yt_1OLI4qiHh9Q", "yt_P39ksl_9EFc"],
    "skincare": ["yt_P2VDMq1mF24", "yt_CYizyqEfk1s", "yt_SKOOSv-9pYQ"],
    "yoga": ["yt_M-s835Zq1nM", "yt_XyaMyzLkOlA"],
    "language": ["yt_gybAlzzE_lU", "yt_E1xsO_hd43s"],
    "makeup": ["yt_YuQW_HSXcBI", "yt_raEnFm0sdnc"],
    "finance": ["yt_37F9l3Q1HzE", "yt_EyBcps1e38o"],
    "guitar": ["yt_9ZXP2mIgK_o", "yt_q4wtHuEj-eA"],
    "gardening": ["yt_6lswAFsS6qs"],
    "math": ["yt_Wkds7WSy0po"],
}

# Reverse lookup: video → category
VID_TO_CAT = {}
for cat, vids in CATEGORIES.items():
    for vid in vids:
        VID_TO_CAT[vid] = cat

# ── Per-video instructional quality (query-independent baseline) ────────────
# 0=not instructional, 1=minimal, 2=moderate, 3=highly instructional
INSTRUCTIONAL_QUALITY = {
    # Cooking
    "yt_6Db60rwdOuY": 2, "yt_j19HFPplMQ0": 1, "yt_f-twQeqQAZE": 3,
    "yt_AmC9SmCBUj4": 3, "yt_dv7BDkQmvhA": 2, "yt_yzDAApWOgYg": 3,
    "yt_VAHnSiSBf5g": 2, "yt_WdYd_UCiZsM": 3, "yt_xxzNu82YL10": 1,
    "yt_-0Zp_fL3jCo": 3, "yt_6ezSIJZt6io": 3, "yt_ayV17R7gE8M": 3,
    # Fitness
    "yt_jx9I-1D6GLs": 3, "yt_LFF7iCW5Y2E": 3, "yt_-KKfHX56OTY": 1,
    "yt_Qn1voplJI4I": 3, "yt_B1KbolzpWD4": 3, "yt_wCPvebDq2yw": 3,
    "yt_xLpbXcIwQpk": 3, "yt__6WERjVqBwE": 1,
    # Crafts
    "yt_OEWtQLQ9czs": 3, "yt_hB-AAGcMsO4": 1, "yt_Y0t4mUr5vFI": 2,
    "yt_fcRPECfjHjg": 2, "yt_TCM_eADif8A": 3, "yt_IkHYli39Exg": 2,
    "yt_UqiLaQXY5tY": 3, "yt_duf4Ra5-0pE": 3, "yt_pgxmPoVyj80": 1,
    # Tech
    "yt_DVJDUAgPKR4": 2, "yt_9csDczj86v8": 3, "yt_zhgGaIOJwBQ": 1,
    "yt_FPe3TAKOT9Y": 2, "yt_tj-tp_XDCqE": 2, "yt_Y2ZG9SrxNo8": 3,
    "yt_3LiFEaPUiNQ": 1, "yt_ka4nR0wqpuY": 0, "yt_jHw5FsWYgBY": 2,
    # History
    "yt_Xv-_c3WtY8M": 2, "yt_ibp1HgWoBPA": 1, "yt_A9ejBOhq-I8": 2,
    "yt_aZw_u60v_5U": 2, "yt_rcPVDigl1_c": 1, "yt_XE_NjAF7Le0": 3,
    # Tiffany
    "yt_sW5QGWCQgHY": 2, "yt_tft9uUJFrkE": 3, "yt_VmRvzOniIdg": 2,
    "yt_xX2ejPGswBM": 1, "yt_6HwBIB5S55U": 2, "yt_HMc3ZGNmF3o": 2,
    "yt_7BPU0F4da5w": 2, "yt_EBEr2g2-DSk": 2, "yt_aGydCvIJKNw": 2,
    # Photography
    "yt_souKz302luU": 2, "yt_L7N8U7emCVU": 3, "yt_l2o-GgFyLBg": 3,
    "yt__Xk3XIyYsO8": 3, "yt_qnRsO7uTFRc": 3,
    # Music
    "yt_QqXWu5O3uyI": 3, "yt_hSFEv6kbC0c": 2, "yt_BG50BcFctyc": 3,
    "yt_JKd8h7fJMvQ": 3,
    # Coding
    "yt_NGlX-1jhYJE": 2, "yt_u5vWFfZ6Csc": 3, "yt_BVyxC68kHdc": 3,
    "yt_cTP6Bm4MXZU": 3, "yt_3fkP88FDgaI": 2,
    # Drawing
    "yt_y6KjkIbz9OY": 3, "yt_1OLI4qiHh9Q": 3, "yt_P39ksl_9EFc": 3,
    # Skincare
    "yt_P2VDMq1mF24": 3, "yt_CYizyqEfk1s": 3, "yt_SKOOSv-9pYQ": 3,
    # Yoga
    "yt_M-s835Zq1nM": 3, "yt_XyaMyzLkOlA": 2,
    # Language
    "yt_gybAlzzE_lU": 3, "yt_E1xsO_hd43s": 3,
    # Makeup
    "yt_YuQW_HSXcBI": 3, "yt_raEnFm0sdnc": 3,
    # Finance
    "yt_37F9l3Q1HzE": 2, "yt_EyBcps1e38o": 3,
    # Guitar
    "yt_9ZXP2mIgK_o": 3, "yt_q4wtHuEj-eA": 3,
    # Gardening
    "yt_6lswAFsS6qs": 3,
    # Math
    "yt_Wkds7WSy0po": 3,
}

# ── Query definitions with relevance rules ──────────────────────────────────
# Each query defines:
#   - relevant_categories: categories where most videos are relevant
#   - specific_overrides: {video_id: (relevance, instructional)} for fine-tuning

QUERIES = [
    # --- Original 10 (updated with expanded pool) ---
    {
        "query": "how to cook steak",
        "relevant_categories": {},  # manual overrides only
        "specific": {
            # Highly relevant: directly about steak
            "yt_f-twQeqQAZE": 3, "yt_AmC9SmCBUj4": 3,
            # Relevant: general cooking skills applicable to steak
            "yt_VAHnSiSBf5g": 2, "yt_yzDAApWOgYg": 2, "yt_6ezSIJZt6io": 2,
            # Marginal: cooking but not steak
            "yt_6Db60rwdOuY": 1, "yt_j19HFPplMQ0": 1, "yt_WdYd_UCiZsM": 1,
            "yt_dv7BDkQmvhA": 1, "yt_xxzNu82YL10": 1,
            "yt_-0Zp_fL3jCo": 1, "yt_ayV17R7gE8M": 1,
        },
    },
    {
        "query": "quick workout for beginners",
        "relevant_categories": {},
        "specific": {
            "yt_jx9I-1D6GLs": 3, "yt_Qn1voplJI4I": 3,
            "yt_LFF7iCW5Y2E": 2, "yt_B1KbolzpWD4": 2,
            "yt_wCPvebDq2yw": 2, "yt_xLpbXcIwQpk": 2,
            "yt_-KKfHX56OTY": 1, "yt__6WERjVqBwE": 1,
        },
    },
    {
        "query": "DIY craft tutorial",
        "relevant_categories": {},
        "specific": {
            "yt_Y0t4mUr5vFI": 3, "yt_TCM_eADif8A": 3,
            "yt_UqiLaQXY5tY": 3, "yt_duf4Ra5-0pE": 3,
            "yt_OEWtQLQ9czs": 2, "yt_fcRPECfjHjg": 2,
            "yt_hB-AAGcMsO4": 1, "yt_IkHYli39Exg": 1,
            "yt_pgxmPoVyj80": 1,
        },
    },
    {
        "query": "computer hardware tips",
        "relevant_categories": {},
        "specific": {
            "yt_tj-tp_XDCqE": 3, "yt_Y2ZG9SrxNo8": 3, "yt_jHw5FsWYgBY": 3,
            "yt_DVJDUAgPKR4": 2, "yt_9csDczj86v8": 2,
            "yt_zhgGaIOJwBQ": 2, "yt_FPe3TAKOT9Y": 2,
            "yt_ka4nR0wqpuY": 1, "yt_3LiFEaPUiNQ": 1,
        },
    },
    {
        "query": "tiffany design meaning",
        "relevant_categories": {},
        "specific": {
            "yt_tft9uUJFrkE": 3, "yt_aGydCvIJKNw": 3,
            "yt_sW5QGWCQgHY": 2, "yt_VmRvzOniIdg": 2,
            "yt_EBEr2g2-DSk": 2, "yt_HMc3ZGNmF3o": 2,
            "yt_xX2ejPGswBM": 1, "yt_7BPU0F4da5w": 1,
            "yt_6HwBIB5S55U": 1,
        },
    },
    {
        "query": "history facts",
        "relevant_categories": {},
        "specific": {
            "yt_Xv-_c3WtY8M": 3, "yt_XE_NjAF7Le0": 3,
            "yt_ibp1HgWoBPA": 2, "yt_A9ejBOhq-I8": 2,
            "yt_aZw_u60v_5U": 2,
            "yt_rcPVDigl1_c": 1,
            # Tiffany history is marginal for general "history facts"
            "yt_sW5QGWCQgHY": 1, "yt_tft9uUJFrkE": 1,
        },
    },
    {
        "query": "knife skills cooking technique",
        "relevant_categories": {},
        "specific": {
            "yt_j19HFPplMQ0": 3, "yt_yzDAApWOgYg": 3,
            "yt_f-twQeqQAZE": 1, "yt_AmC9SmCBUj4": 1,
            "yt_VAHnSiSBf5g": 1,
            "yt_-0Zp_fL3jCo": 1, "yt_6ezSIJZt6io": 1,
        },
    },
    {
        "query": "home exercise no equipment",
        "relevant_categories": {},
        "specific": {
            "yt_LFF7iCW5Y2E": 3, "yt_xLpbXcIwQpk": 3,
            "yt_jx9I-1D6GLs": 2, "yt_Qn1voplJI4I": 2, "yt_B1KbolzpWD4": 2,
            "yt__6WERjVqBwE": 2,
            "yt_-KKfHX56OTY": 1, "yt_wCPvebDq2yw": 1,
        },
    },
    {
        "query": "paper craft ideas",
        "relevant_categories": {},
        "specific": {
            "yt_Y0t4mUr5vFI": 3,
            "yt_TCM_eADif8A": 2, "yt_UqiLaQXY5tY": 2,
            "yt_duf4Ra5-0pE": 2, "yt_pgxmPoVyj80": 2,
            "yt_OEWtQLQ9czs": 1, "yt_hB-AAGcMsO4": 1,
        },
    },
    {
        "query": "tiffany jewelry history",
        "relevant_categories": {},
        "specific": {
            "yt_tft9uUJFrkE": 3, "yt_sW5QGWCQgHY": 3,
            "yt_EBEr2g2-DSk": 3, "yt_6HwBIB5S55U": 3,
            "yt_VmRvzOniIdg": 2, "yt_7BPU0F4da5w": 2,
            "yt_HMc3ZGNmF3o": 2, "yt_aGydCvIJKNw": 2,
            "yt_xX2ejPGswBM": 1,
        },
    },
    # --- 5 new queries ---
    {
        "query": "beginner photography tips",
        "relevant_categories": {},
        "specific": {
            "yt_L7N8U7emCVU": 3, "yt__Xk3XIyYsO8": 3, "yt_l2o-GgFyLBg": 3,
            "yt_souKz302luU": 2, "yt_qnRsO7uTFRc": 2,
        },
    },
    {
        "query": "learn to code python",
        "relevant_categories": {},
        "specific": {
            "yt_BVyxC68kHdc": 3, "yt_u5vWFfZ6Csc": 3, "yt_cTP6Bm4MXZU": 3,
            "yt_NGlX-1jhYJE": 2, "yt_3fkP88FDgaI": 2,
        },
    },
    {
        "query": "how to draw faces",
        "relevant_categories": {},
        "specific": {
            "yt_P39ksl_9EFc": 3,
            "yt_y6KjkIbz9OY": 2, "yt_1OLI4qiHh9Q": 2,
        },
    },
    {
        "query": "skincare routine for beginners",
        "relevant_categories": {},
        "specific": {
            "yt_CYizyqEfk1s": 3, "yt_SKOOSv-9pYQ": 3, "yt_P2VDMq1mF24": 3,
        },
    },
    {
        "query": "music production tutorial",
        "relevant_categories": {},
        "specific": {
            "yt_QqXWu5O3uyI": 3, "yt_BG50BcFctyc": 3,
            "yt_JKd8h7fJMvQ": 3,
            "yt_hSFEv6kbC0c": 2,
        },
    },
]


def generate():
    """Generate ground truth JSON with dual labels."""
    output = {
        "description": (
            "Ground truth with dual labels for 15 queries across 86 videos. "
            "relevance: 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant. "
            "instructional: 0=not instructional, 1=minimal, 2=moderate, 3=highly instructional."
        ),
        "queries": [],
    }

    for q_def in QUERIES:
        labels = {}
        for vid in LABELED_VIDEOS:
            relevance = q_def["specific"].get(vid, 0)
            instructional = INSTRUCTIONAL_QUALITY.get(vid, 0)
            labels[vid] = {"relevance": relevance, "instructional": instructional}

        output["queries"].append({
            "query": q_def["query"],
            "labels": labels,
        })

    return output


def main():
    gt = generate()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    # Summary
    n_queries = len(gt["queries"])
    n_videos = len(LABELED_VIDEOS)
    for q in gt["queries"]:
        n_rel = sum(1 for v in q["labels"].values() if v["relevance"] >= 2)
        n_instr = sum(1 for v in q["labels"].values() if v["instructional"] >= 2)
        print(f"  \"{q['query']}\": {n_rel} relevant, {n_instr} instructional")
    print(f"\nGenerated {out_path}: {n_queries} queries × {n_videos} videos")


if __name__ == "__main__":
    main()
