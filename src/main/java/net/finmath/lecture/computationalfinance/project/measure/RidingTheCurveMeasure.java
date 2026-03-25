package net.finmath.lecture.computationalfinance.project.measure;

import net.finmath.exception.CalculationException;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;

/**
 * Real-world "riding the yield curve" measure for the LIBOR Market Model.
 *
 * <p>
 * Under this measure the drift of forward rate \( L_j \) is chosen so that
 * the entire initial term structure propagates forward by one period at every
 * simulation step. Let \( m = m(t) \) denote the index of the current period
 * (i.e.\ {@code firstForwardRateIndex}). The drift at simulation time \( t \) is:
 * \[
 *   \mu_j(t) = \frac{X_{j-m}(0) - X_{j-m+1}(0)}{\delta_{m-1}},
 *   \qquad X_i(0) = f^{-1}(L_i(0)),
 * \]
 * where \( f^{-1} \) is the inverse state-space transform,
 * \( L_i(0) \) is the initial forward rate for period \( i \),
 * and \( \delta_{m-1} \) is the length of the period preceding the current one.
 *
 * <p>
 * Consequently, after \( k \) time steps:
 * \[
 *   \mathbb{E}[L_j(T_k)] \approx L_{j-k}(0),
 * \]
 * meaning the yield curve is "ridden" forward in time — rate \( L_j \)
 * converges toward the initial short-end level as time passes.
 *
 * <p>
 * The numeraire is set to 1 so that {@code product.getValue(simulation)} returns
 * the raw (undiscounted) expected cashflow under \( \mathbb{P} \).
 *
 * @author Felipe, GM-1, GM-2
 * @see Measure
 * @see ZeroDriftMeasure
 */
public class RidingTheCurveMeasure implements Measure {

	private final ForwardCurve forwardCurve;

	/**
	 * Creates a RidingTheCurveMeasure from the initial forward curve.
	 *
	 * @param forwardCurve The initial forward curve \( L_j(0) \).
	 */
	public RidingTheCurveMeasure(final ForwardCurve forwardCurve) {
		this.forwardCurve = forwardCurve;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * Let \( m = \texttt{firstForwardRateIndex} \). For each live component
	 * \( j \geq m \) the drift is:
	 * \[
	 *   \mu_j(t) = \frac{X_{j-m}(0) - X_{j-m+1}(0)}{\delta_{m-1}},
	 *   \qquad X_i(0) = f^{-1}(L_i(0)),
	 * \]
	 * where \( L_i(0) \) is looked up from the initial forward curve and
	 * \( \delta_{m-1} \) is the period length of the preceding tenor interval.
	 * The initial internal states \( X_i(0) \) are obtained via
	 * {@link StateSpaceTransform#getInitialState(double, int)}.
	 */
	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final int firstForwardRateIndex,
			final RandomVariable[] realizationAtTimeIndex,
			final StateSpaceTransform stateSpaceTransform,
			final LIBORMarketModel model) {

		final int n = model.getNumberOfComponents();
		final RandomVariable[] drift = new RandomVariable[n];

        // Let k = firstForwardRateIndex, i. e. T_{k-1} <= t_{timeIndex} < T_k.
		// We want mu_j(t) = (L_{j-k}(0) - L_{j-(k-1)}(0)) / delta_{k-1} for T_{k-1} <= t < T_k
		for(int j = firstForwardRateIndex; j < n; j++) {
			final double periodLength = model.getLiborPeriodDiscretization().getTimeStep(firstForwardRateIndex-1);

			final double rateNew = forwardCurve.getForward(null, model.getLiborPeriod(j-firstForwardRateIndex), model.getLiborPeriodDiscretization().getTimeStep(j-firstForwardRateIndex));
			final double stateNew = stateSpaceTransform.getInitialState(rateNew, j-firstForwardRateIndex); // L_{j-k}(0)

			final double stateOld; // L_{j-(k-1)}(0)
			if(firstForwardRateIndex > 0) { // which should always be the case
				final double rateOld = forwardCurve.getForward(null, model.getLiborPeriod(j-firstForwardRateIndex+1), model.getLiborPeriodDiscretization().getTimeStep(j-firstForwardRateIndex+1));
				stateOld = stateSpaceTransform.getInitialState(rateOld, j - firstForwardRateIndex + 1);
			} else {
				stateOld = 0; // no predecessor
			}

			drift[j] = model.getRandomVariableForConstant((stateNew - stateOld) / periodLength);
		}
		return drift;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>Returns 1.0 so that pricing products compute raw expected cashflows
	 * without discounting.
	 */
	@Override
	public RandomVariable getNumeraire(
			final MonteCarloProcess process,
			final int liborTimeIndex,
			final LIBORMarketModel model) throws CalculationException {
		return model.getRandomVariableForConstant(1.0);
	}
}
