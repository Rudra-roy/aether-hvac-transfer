import json


def _summary(path, total, comfort, energy, carbon, steps=8640):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({
        "total_reward": total, "total_steps": steps,
        "total_normalized_productivity_regret": comfort, "total_normalized_energy_cost": energy,
        "total_normalized_carbon_emission": carbon}))


def test_decomposition_sums_to_score(tmp_path):
    from harness.anatomy import decompose
    # reward = -(0.6 |comfort| + 0.2 energy + 0.2 carbon) summed over steps
    for i, (b, w) in enumerate([("building_a", "fairbanks_1"), ("building_b", "phoenix_2")]):
        c, e, k = -100.0 * (i + 1), 300.0, 200.0
        total = -(0.6 * abs(c) + 0.2 * e + 0.2 * k)
        _summary(tmp_path / b / w / "pid_water", total, c, e, k)
    d = decompose(tmp_path, "pid_water")
    assert set(d["by_station"]) == {"fairbanks", "phoenix"}
    for row in [d["mean"], *d["by_station"].values()]:
        assert abs(row["comfort"] + row["energy"] + row["carbon"] - row["score"]) < 1e-12
    # scenario totals: 0.6*100+0.2*300+0.2*200 = 160 and 0.6*200+60+40 = 220
    assert abs(d["mean"]["score"] - (160 + 220) / 2 / 8640) < 1e-12
