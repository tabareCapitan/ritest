import pandas as pd

from ritest import RitestResult, ritest


def _toy_df():
    """
    Small, design-consistent dataset usable for:
    - simple RI
    - stratified RI
    - clustered RI
    - cluster-in-strata RI
    - linear + generic statistic paths

    Treatment is constant within clusters (required by permute_assignment).
    Clusters are nested inside strata.
    """

    return pd.DataFrame(
        {
            # outcome (varied)
            "y": [2.1, 2.5, 3.0, 3.8, 4.2, 4.9, 5.1, 5.3, 6.0, 6.5],
            # two strata: 0 and 1
            "strata": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            # five clusters, nested cleanly inside strata
            # strata 0: clusters 1, 2
            # strata 1: clusters 3, 4, 5
            "cluster": [1, 1, 2, 2, 2, 3, 3, 4, 5, 5],
            # treatment constant within each cluster (required!)
            # clusters: 1→0, 2→1, 3→0, 4→1, 5→1
            "treat": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
            # a covariate
            "x": [0.1, 0.2, 0.25, 0.3, 0.31, 0.5, 0.55, 0.6, 0.65, 0.7],
        }
    )


def test_ritest_linear_smoke():
    df = _toy_df()
    res = ritest(
        df=df,
        permute_var="treat",
        formula="y ~ treat + x",
        stat="treat",
        reps=200,
        ci_mode="none",
        seed=123,
    )

    assert isinstance(res, RitestResult)
    assert 0.0 <= res.pval <= 1.0
    assert res.reps == 200
    assert res.alternative == "two-sided"
    assert res.stratified is False
    assert res.clustered is False
    assert res.weights is False

    # summary and explain should return non-empty strings
    s = res.summary(print_out=False)
    e = res.explain()
    assert isinstance(s, str) and s.strip()
    assert isinstance(e, str) and e.strip()


def test_ritest_stat_fn_smoke():
    df = _toy_df()

    def my_stat(d):
        return d["y"].corr(d["treat"])

    res = ritest(
        df=df,
        permute_var="treat",
        stat_fn=my_stat,
        reps=200,
        ci_mode="none",
        seed=123,
    )

    assert isinstance(res, RitestResult)
    assert 0.0 <= res.pval <= 1.0
    assert res.reps == 200
    assert res.alternative == "two-sided"
    # generic path may or may not store perm_stats depending on config;
    # here we only check the attribute exists.
    assert hasattr(res, "perm_stats")


def test_ritest_strata_and_cluster_flags():
    df = _toy_df()
    res = ritest(
        df=df,
        permute_var="treat",
        formula="y ~ treat + x",
        stat="treat",
        strata="strata",
        cluster="cluster",
        reps=100,
        ci_mode="none",
        seed=123,
    )

    assert isinstance(res, RitestResult)
    assert res.stratified is True
    assert res.clustered is True
    # CI mode none should gate out coef CIs
    assert res.coef_ci_bounds is None
    assert res.coef_ci_band is None


def test_ritest_requires_exactly_one_of_stat_or_stat_fn():
    df = _toy_df()

    # Neither stat nor stat_fn -> error
    try:
        ritest(df=df, permute_var="treat")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError when neither stat nor stat_fn is provided."
        )

    # Both stat and stat_fn -> error
    def my_stat(d):
        return d["y"].mean()

    try:
        ritest(
            df=df,
            permute_var="treat",
            formula="y ~ treat + x",
            stat="treat",
            stat_fn=my_stat,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError when both stat and stat_fn are provided."
        )
