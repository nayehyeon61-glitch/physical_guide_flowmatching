"""scripts/local_era5_to_kai.py --src-daily: a daily-mean archive in several layouts gives the same daily file as
the hourly archive whose four snapshots average to it (the plumbing: variable, level, day, orientation)."""
import os

import netCDF4 as nc
import numpy as np
import pytest

from pgcfm import contract
from scripts import local_era5_to_kai as L

LAT = np.linspace(90, -90, 37)  # N -> S, 5 deg
LON = 5.0 * np.arange(72)
LEV = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
UPPER = ["z", "q", "t", "u", "v"]
SURF = ["t2m", "sp", "tisr", "tcwv", "mslp", "skt", "u10", "v10"]
LONG = {"z": "geopotential", "q": "specific_humidity", "t": "temperature", "u": "u_component_of_wind",
        "v": "v_component_of_wind", "t2m": "2m_temperature", "sp": "surface_pressure",
        "tcwv": "total_column_water_vapour", "mslp": "mean_sea_level_pressure"}


def field(name, k, day):
    """A smooth field that tells variable, level and day apart."""
    i = (UPPER + SURF).index(name)
    return (100.0 * i + 10.0 * k + day + np.cos(np.deg2rad(LAT))[:, None] * np.sin(np.deg2rad(LON + 20.0 * i))[None]
            + 0.01 * LAT[:, None]).astype(np.float32)


def day_of(year, i):
    return np.datetime64(f"{year}-01-01") + np.timedelta64(int(i), "D")


def axes(d, lat, levels=LEV, lev_name="level"):
    d.createDimension("latitude", len(lat))
    d.createDimension("longitude", len(LON))
    d.createVariable("latitude", "f8", ("latitude",))[:] = lat
    d.createVariable("longitude", "f8", ("longitude",))[:] = LON
    if levels is not None:
        d.createDimension(lev_name, len(levels))
        d.createVariable(lev_name, "f8", (lev_name,))[:] = levels


def write_hourly(root, year, n):
    os.makedirs(root, exist_ok=True)
    off = np.array([1.0, -1.0, 2.0, -2.0], np.float32)  # the four snapshots average to the daily field
    for i in range(n):
        with nc.Dataset(os.path.join(root, str(day_of(year, i)).replace("-", "") + ".nc"), "w") as d:
            axes(d, LAT)
            d.createDimension("time", 4)
            t = d.createVariable("time", "f8", ("time",))
            t.units = f"hours since {day_of(year, i)} 00:00:00"
            t[:] = [0, 6, 12, 18]
            for v in UPPER:
                x = np.stack([field(v, k, i) for k in range(len(LEV))])
                d.createVariable(v, "f4", ("time", "level", "latitude", "longitude"))[:] = x[None] + off[:, None, None, None]
            for v in SURF:
                d.createVariable(v, "f4", ("time", "latitude", "longitude"))[:] = field(v, 0, i)[None] + off[:, None, None]


def write_daily_a(root, year, n, sub=lambda day: ""):
    """One file per day with everything, a time axis of length 1."""
    for i in range(n):
        day = day_of(year, i)
        os.makedirs(os.path.join(root, sub(day)), exist_ok=True)
        with nc.Dataset(os.path.join(root, sub(day), str(day).replace("-", "") + ".nc"), "w") as d:
            axes(d, LAT)
            d.createDimension("time", 1)
            t = d.createVariable("time", "f8", ("time",))
            t.units = "hours since 1900-01-01 00:00:00"
            t[:] = [(day - np.datetime64("1900-01-01")) / np.timedelta64(1, "h")]
            for v in UPPER:
                d.createVariable(v, "f4", ("time", "level", "latitude", "longitude"))[0] = np.stack(
                    [field(v, k, i) for k in range(len(LEV))])
            for v in SURF[:-2]:
                d.createVariable(v, "f4", ("time", "latitude", "longitude"))[0] = field(v, 0, i)


def write_daily_b(root, year, n):
    """Year folders, no time axis, levels in Pa in reverse order, latitude S -> N, long names, no skt, no tisr."""
    order = list(range(len(LEV)))[::-1]
    for i in range(n):
        day = day_of(year, i)
        os.makedirs(os.path.join(root, str(year)), exist_ok=True)
        with nc.Dataset(os.path.join(root, str(year), f"era5_daily_{day}.nc"), "w") as d:
            axes(d, LAT[::-1], [100.0 * LEV[k] for k in order], "plev")
            for v in UPPER:
                d.createVariable(LONG[v], "f4", ("plev", "latitude", "longitude"))[:] = np.stack(
                    [field(v, k, i)[::-1] for k in order])
            for v in ("t2m", "sp", "tcwv", "mslp"):
                d.createVariable(LONG[v], "f4", ("latitude", "longitude"))[:] = field(v, 0, i)[::-1]


def write_daily_c(root, year, n):
    """One file per variable and year, the days on a time axis (ERA5 CDS names of the axes)."""
    os.makedirs(root, exist_ok=True)
    for v in UPPER + SURF[:-2]:
        with nc.Dataset(os.path.join(root, f"{v}_{year}.nc"), "w") as d:
            axes(d, LAT, LEV if v in UPPER else None, "pressure_level")
            d.createDimension("valid_time", n)
            t = d.createVariable("valid_time", "f8", ("valid_time",))
            t.units = f"days since {year}-01-01 12:00:00"
            t[:] = np.arange(n)
            name = "msl" if v == "mslp" else v
            if v in UPPER:
                d.createVariable(name, "f4", ("valid_time", "pressure_level", "latitude", "longitude"))[:] = np.stack(
                    [np.stack([field(v, k, i) for k in range(len(LEV))]) for i in range(n)])
            else:
                d.createVariable(name, "f4", ("valid_time", "latitude", "longitude"))[:] = np.stack(
                    [field(v, 0, i) for i in range(n)])


def constants(root):
    os.makedirs(root, exist_ok=True)
    for name, var in (("orog", "z"), ("lsm", "lsm")):
        with nc.Dataset(os.path.join(root, f"{name}.nc"), "w") as d:
            axes(d, LAT, None)
            d.createDimension("time", 1)
            d.createVariable(var, "f4", ("time", "latitude", "longitude"))[0] = field("t2m", 0, 0)
    return ["--orog", os.path.join(root, "orog.nc"), "--lsm", os.path.join(root, "lsm.nc")]


def convert(src, out, n, const, extra=()):
    with pytest.raises(SystemExit) as e:  # --check stops the program: 0 = the archive can be read
        L.main(["--src", src, "--out", out, "--years", "2015", "--check", *extra, *const])
    assert e.value.code == 0
    L.main(["--src", src, "--out", out, "--years", "2015", "--daily-mean", "--days", str(n), "--workers", "1",
            *extra, *const])
    return np.load(os.path.join(out, "kaiday", "2015.npy"))


def test_daily_archive_layouts_match_the_hourly_archive(tmp_path):
    tmp, n = str(tmp_path), 3
    const = constants(os.path.join(tmp, "const"))
    write_hourly(os.path.join(tmp, "hourly"), 2015, n)
    ref = convert(os.path.join(tmp, "hourly"), os.path.join(tmp, "out_h"), n, const)
    assert ref.shape == (n, contract.N_CHANNELS, 32, 64)
    write_daily_a(os.path.join(tmp, "a"), 2015, n)
    a = convert(os.path.join(tmp, "a"), os.path.join(tmp, "out_a"), n, const, ["--src-daily"])
    assert np.allclose(a, ref, rtol=0, atol=2e-4)
    # the channels are what they say: z500 of day 2 is variable 0, level index 7
    w = L.weights(LAT, LON)
    assert np.allclose(a[2, contract.CHANNEL_INDEX["z500"]], L.regrid(field("z", 7, 2), *w), atol=1e-4)
    write_daily_c(os.path.join(tmp, "c"), 2015, n)
    c = convert(os.path.join(tmp, "c"), os.path.join(tmp, "out_c"), n, const, ["--src-daily"])
    assert np.allclose(c, a, rtol=0, atol=1e-5)
    write_daily_b(os.path.join(tmp, "b"), 2015, n)
    b = convert(os.path.join(tmp, "b"), os.path.join(tmp, "out_b"), n, const, ["--src-daily"])
    filled = [contract.CHANNEL_INDEX["skt"], contract.CHANNEL_INDEX["tisr"]]
    same = [i for i in range(contract.N_CHANNELS) if i not in filled]
    assert np.allclose(b[:, same], a[:, same], rtol=0, atol=1e-5)
    assert np.array_equal(b[:, filled[0]], b[:, contract.CHANNEL_INDEX["t2m"]])  # skt := t2m
    tisr = b[:, filled[1]]
    assert np.isfinite(tisr).all() and tisr.min() >= 0 and tisr[0, -1].max() == 0  # January: polar night in the north
    assert tisr[0, 2].mean() > tisr[0, 20].mean() > 0  # more sun at 76 S than at 26 N in the southern summer


def test_month_folders_and_a_missing_day(tmp_path):
    tmp, n = str(tmp_path), 33  # into February: the pattern of January's folder does not fit
    const = constants(os.path.join(tmp, "const"))
    write_daily_a(os.path.join(tmp, "d"), 2015, n, sub=lambda day: os.path.join(str(day)[:4], str(day)[5:7]))
    d = convert(os.path.join(tmp, "d"), os.path.join(tmp, "out_d"), n, const, ["--src-daily"])
    w = L.weights(LAT, LON)
    for i in (0, 30, 31, 32):
        assert np.allclose(d[i, contract.CHANNEL_INDEX["t850"]], L.regrid(field("t", 10, i), *w), atol=1e-4)
    os.remove(os.path.join(tmp, "d", "2015", "01", "20150105.nc"))
    with pytest.raises(Exception, match="2015-01-05"):
        L.year_job(os.path.join(tmp, "d"), os.path.join(tmp, "out_e"), 2015, d[0, contract.CHANNEL_INDEX["orog"]],
                   "auto", "5.625", True, 8, True)


def test_check_says_not_ok_for_an_empty_folder(tmp_path, capsys):
    os.makedirs(os.path.join(str(tmp_path), "empty", "sub"))
    assert not L.check_daily(os.path.join(str(tmp_path), "empty"), 2015, "auto")
    assert "NOT OK" in capsys.readouterr().out
