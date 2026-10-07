# Interest Rate Models: LIBOR Market Model Plug-ins

A Monte Carlo LIBOR Market Model in Java, built on [finmath-lib](https://github.com/finmath/finmath-lib). Two design decisions that are normally hard-coded, the **probability measure** and the **state-space transform**, are refactored into plug-in interfaces. New measures and new rate dynamics can then be added without changing the model class (the open/closed principle).

Group project for **Computational Finance and its Object Oriented Implementation** at **LMU Munich** (Prof. Christian Fries), winter term 2025/26.

## What's implemented

| Task | Topic | Classes |
| --- | --- | --- |
| Ex 1 + 2 | **T_k-forward measure** and a `Measure` plug-in interface that replaces the hard-coded SPOT/TERMINAL enum | `Measure`, `SpotMeasure`, `TerminalMeasure`, `ForwardMeasure` |
| Ex 3 | **Real-world measure P**: a zero-drift (martingale) measure and a "riding the yield curve" measure | `ZeroDriftMeasure`, `RidingTheCurveMeasure` |
| Ex 5 | **Displaced lognormal model** with a per-rate displacement a_j, which allows negative rates down to −a_j | `DisplacedLogNormalStateSpaceTransform` |
| Ex 6 | **State-space transform plug-in** that replaces the hard-coded NORMAL/LOGNORMAL enum | `StateSpaceTransform`, `NormalStateSpaceTransform`, `LogNormalStateSpaceTransform` |

The model class `AugmentedLIBORMarketModel` is derived from finmath-lib's `LIBORMarketModelFromCovarianceModel`. A `Measure` and a `StateSpaceTransform` are injected into it at construction time.

### Design notes

- **Measure:** a measure determines the drift of each forward rate and the numeraire. Spot, terminal and T_k-forward measures differ only in the summation range and sign of the drift term, so the drift logic belongs to the measure.
- **State-space transform:** in addition to the forward and inverse transform L = f(X), the transform supplies the Itô correction and the drift weight factor g(L). Keeping all three in one object (the Strategy pattern) means an incompatible combination, such as a lognormal transform without its Itô correction, cannot be configured.
- **Per-rate parameters:** every transform method receives the LIBOR index, so a transform can use a different parameter for each rate. The displaced model uses this for its per-rate displacement.

## Tests

[`LIBORMarketModelPluginTest`](src/test/java/net/finmath/lecture/computationalfinance/project/LIBORMarketModelPluginTest.java) contains one test per task. Each test prices in-the-money, at-the-money and out-of-the-money caplets by Monte Carlo and compares them with analytic values.

- **`ex1and2_forwardMeasure`:** caplets under T_k-forward measures (k = 6, 10, 14, 18), with normal and lognormal dynamics.
- **`ex3_realWorldMeasure`:** checks E[L_j(T_k)] = L_j(0) under zero drift, and E[L_j(T_k)] ≈ L_{j−k}(0) under riding the curve.
- **`ex5_displacedLognormal`:** displaced lognormal caplets under the spot and terminal measures, compared with the Black formula for the shifted rate.
- **`ex6_stateSpaceTransform`:** checks that the plug-in version matches the original enum-based implementation bit for bit (same seed, zero error), and compares normal and lognormal prices.

## Build and run

Requires Java 17 and Maven.

```bash
mvn clean test            # run the unit tests
mvn javadoc:javadoc       # API docs (LaTeX math in Javadoc) -> target/site/apidocs
mvn checkstyle:check      # finmath code style
```

## Team

Developed in a team of three (GM = group member):

- [Felipe](https://github.com/feroe-dev): measure plug-in interface and the spot, terminal and forward measures (Ex 1 + 2), the real-world measures (Ex 3), and Javadoc
- GM-1: forward-measure case distinctions, the displaced lognormal model (Ex 5), drift caching, and tests
- GM-2: state-space transform plug-in (Ex 6) and comparison tests

The commit history shows each person's individual contributions.

## Acknowledgements

The project template, `LectureProjectData` and the build configuration were provided by the lecture (Christian Fries, quantLab). The model is based on [finmath-lib](https://github.com/finmath/finmath-lib) (Apache License 2.0).

## License

`AugmentedLIBORMarketModel.java` is a modified version of `LIBORMarketModelFromCovarianceModel` from finmath-lib, © Christian P. Fries. It is distributed under the Apache License, Version 2.0. The licence text and finmath-lib's NOTICE are in [`licenses/finmath-lib/`](licenses/finmath-lib/), and the file's header and Javadoc describe the changes.
