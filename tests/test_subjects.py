"""Subject-bucket logic tests — normalization, dominant subject, fuzzy bucket planning."""
from __future__ import annotations

from collections import Counter

from geolip_anima_trainer import subject_buckets as S


def test_normalize_head_noun_and_singularize():
    n = S.normalize_subject
    assert n("Fire Truck") == "truck"
    assert n("the Police Officers") == "officer"
    assert n("tea cup") == "cup"
    assert n("trees") == "tree"
    assert n("women") == "woman"
    assert n("obstacles") == "obstacle"
    assert n("") is None
    assert n(None) is None


def test_normalize_full_phrase_mode():
    assert S.normalize_subject("Fire Truck", head_noun=False) == "fire truck"
    assert S.normalize_subject("the Police Officers", head_noun=False) == "police officer"


def test_dominant_subject_from_vlm_json():
    j = '{"subjects":[{"name":"man","attributes":[]},{"name":"car"}],"actions":[],"setting":"outdoor"}'
    assert S.dominant_subject(j) == "man"
    assert S.dominant_subject('{"subjects":[]}') is None
    assert S.dominant_subject("not json") is None


def test_real_caption_filters_sentinels():
    assert S.real_caption('{"subjects":[]}') == '{"subjects":[]}'
    assert S.real_caption("") is None
    assert S.real_caption("__PARSEFAIL__") is None
    assert S.real_caption("__NO_TAGS__") is None
    assert S.real_caption(None) is None


def test_plan_buckets_merges_small_into_similar():
    cfg = S.SubjectBucketConfig(min_bucket_size=3, fuzzy_cutoff=0.6, drop_unmergeable=True)
    # 'truck' is big (4); 'trucks' is a small near-duplicate -> should merge into 'truck'
    subjects = ["truck"] * 4 + ["car"] * 3 + ["trucks"] * 1 + ["zebra"] * 1
    plan = S.plan_buckets(subjects, cfg)
    assert plan.mapping["truck"] == "truck"
    assert plan.mapping["car"] == "car"
    assert plan.mapping["trucks"] == "truck"      # fuzzy-merged
    assert plan.mapping["zebra"] is None          # unmergeable + tiny -> dropped


def test_plan_buckets_misc_when_not_dropping():
    cfg = S.SubjectBucketConfig(min_bucket_size=3, fuzzy_cutoff=0.9, drop_unmergeable=False)
    subjects = ["man"] * 5 + ["xyzzy"] * 1
    plan = S.plan_buckets(subjects, cfg)
    assert plan.mapping["man"] == "man"
    assert plan.mapping["xyzzy"] == "misc"


# ---- semantic planner (stub sim_fn — no model/download) --------------------
_SIM = {("dancer", "ballerina"): 0.85, ("dancer", "performer"): 0.70,
        ("ballerina", "performer"): 0.70, ("dancer", "player"): 0.60,
        ("ballerina", "player"): 0.60, ("truck", "car"): 0.80}


def _stub_sim(query, cands):
    import numpy as np
    m = np.zeros((len(query), len(cands)), "float32")
    for i, q in enumerate(query):
        for j, c in enumerate(cands):
            m[i, j] = 1.0 if q == c else (_SIM.get((q, c)) or _SIM.get((c, q)) or 0.0)
    return m


def _sem_cfg(**kw):
    base = dict(min_bucket_size=20, human_min_size=10, min_final_group_size=5,
                sim_threshold=0.45, human_sim_threshold=0.55, keep_small=True)
    base.update(kw)
    return S.SubjectBucketConfig(**base)


def _counts():
    from collections import Counter
    return Counter({"man": 50, "dancer": 3, "ballerina": 3, "truck": 3, "car": 3, "zebra": 2})


def test_semantic_protects_big_and_groups_humans_separately():
    plan = S.plan_buckets_semantic(_counts(), _sem_cfg(), sim_fn=_stub_sim)
    m = plan.mapping
    assert m["man"] == "man"                                   # protected big, untouched
    assert m["dancer"] == m["ballerina"] and m["dancer"].startswith("grp_h_")  # grouped humans
    assert m["dancer"] != "man"                                # never folded into man


def test_semantic_objects_group_and_never_cross_human_boundary():
    plan = S.plan_buckets_semantic(_counts(), _sem_cfg(), sim_fn=_stub_sim)
    m = plan.mapping
    assert m["truck"] == m["car"] and m["truck"].startswith("grp_")
    assert not m["truck"].startswith("grp_h_")                 # object side, not human
    assert m["truck"] != m["dancer"]                           # human != object group


def test_semantic_keep_small_pools_leftovers_not_dropped():
    plan = S.plan_buckets_semantic(_counts(), _sem_cfg(keep_small=True), sim_fn=_stub_sim)
    assert plan.mapping["zebra"] == "misc_other"               # weighted, not None
    plan2 = S.plan_buckets_semantic(_counts(), _sem_cfg(keep_small=False), sim_fn=_stub_sim)
    assert plan2.mapping["zebra"] is None                      # drop when asked


def test_link_or_copy_idempotent(tmp_path):
    # re-running an extraction must not crash on an existing hardlinked variant
    src = tmp_path / "img.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 40)
    dst = tmp_path / "img__anime.png"
    S._link_or_copy(src, dst)                     # first run: create the variant
    assert dst.read_bytes() == src.read_bytes()
    S._link_or_copy(src, dst)                     # second run: must overwrite, not raise
    assert dst.read_bytes() == src.read_bytes()


def test_dampened_repeats_curve():
    from geolip_anima_trainer.build_multiconcept_dataset import dampened_repeats as dr
    assert dr(340, 340) == 1          # biggest concept ~1x
    assert dr(120, 340) == 2
    assert dr(5, 340) == 8            # sparse: bounded at max_repeats, NOT 50x
    assert dr(5, 340, alpha=0.0, max_repeats=50) == 50  # legacy equalize would overtrain


# ---- attribute splitting + data-dependent cap ------------------------------
def test_max_bucket_size_tiers():
    assert S.max_bucket_size(50_000) == 1000
    assert S.max_bucket_size(5_000) == 500
    assert S.max_bucket_size(500) == 250
    assert S.max_bucket_size(500, override=42) == 42


def test_normalize_attr():
    assert S.normalize_attr("Blonde Hair") == "blonde_hair"
    assert S.normalize_attr("long_hair") == "long_hair"
    assert S.normalize_attr("1girl") is None          # count/meta tag dropped
    assert S.normalize_attr("") is None


def test_extract_features_prefers_animetimm():
    vlm = '{"subjects":[{"name":"woman","attributes":["red dress"]},{"name":"car"}]}'
    anime = '{"subjects":[{"name":"1girl","attributes":["blonde_hair","1girl"]}]}'
    r = S.extract_features(vlm, anime, prefer="animetimm")
    assert r.vlm_subject == "woman" and r.anime_subject == "1girl"
    assert r.attrs == ("blonde_hair",)                # animetimm attrs win; 1girl dropped
    assert r.secondary == "car"                        # vlm subjects[1] (animetimm had none)
    # bare-string subject -> no attributes
    assert S.extract_features('{"subjects":["man"]}', None).attrs == ()


def _plan(mapping, counts):
    from collections import Counter
    return S.BucketPlan(mapping=mapping, raw_counts=Counter(counts), actions=[])


def test_split_partitions_by_rarest_attribute():
    cfg = S.SubjectBucketConfig(max_bucket_size=3, attr_min_split=1)
    recs = {f"m{i}": S.ImageRecord("man", None, ("hat",), None) for i in range(4)}
    recs["m4"] = S.ImageRecord("man", None, ("monocle",), None)
    recs["m5"] = S.ImageRecord("man", None, ("monocle", "hat"), None)
    ov, sub, over = S.split_oversized_buckets(recs, _plan({"man": "man"}, {"man": 6}),
                                              cfg, stream="vlm", subject_attr="vlm_subject")
    assert {k[0] for k in ov} == set(recs)             # every image assigned (partition)
    assert all(c <= 3 for c in sub.values())           # no sub-bucket exceeds the cap
    assert over and over[0][0] == "man"


def test_split_secondary_fallback_and_chunk():
    cfg = S.SubjectBucketConfig(max_bucket_size=2, attr_min_split=1)
    # no attributes -> secondary subject; all same secondary + over cap -> even-chunk
    recs = {f"m{i}": S.ImageRecord("man", None, (), "dog") for i in range(5)}
    ov, sub, _ = S.split_oversized_buckets(recs, _plan({"man": "man"}, {"man": 5}),
                                           cfg, stream="vlm", subject_attr="vlm_subject")
    names = set(ov.values())
    assert all("with_dog" in n for n in names)         # secondary-subject differentiation
    assert all(c <= 2 for c in sub.values())           # chunked to <= cap


def test_split_leaves_small_buckets_untouched():
    cfg = S.SubjectBucketConfig(max_bucket_size=10)
    recs = {f"m{i}": S.ImageRecord("man", None, ("hat",), None) for i in range(4)}
    ov, sub, over = S.split_oversized_buckets(recs, _plan({"man": "man"}, {"man": 4}),
                                              cfg, stream="vlm", subject_attr="vlm_subject")
    assert ov == {} and sub == {} and over == []       # under cap -> no overrides
