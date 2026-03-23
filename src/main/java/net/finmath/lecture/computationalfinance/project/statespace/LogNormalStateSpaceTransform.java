package net.finmath.lecture.computationalfinance.project.statespace;

import net.finmath.stochastic.RandomVariable;

/**
 * Log-normal (Black) state-space transform for the LIBOR Market Model.
 *
 * <p>
 * Under the log-normal state space, the natural logarithm of each forward rate
 * \( L_j \) is simulated:
 * \[
 *   X_j = \log L_j, \qquad f(X) = e^X.
 * \]
 * Consequently:
 * <ul>
 *   <li>The initial state is the log of the market rate: \( X_0 = \log L_0 \).</li>
 *   <li>The state-space transform is the exponential: \( L = e^X \).</li>
 *   <li>The inverse transform is the natural logarithm: \( X = \log L \).</li>
 *   <li>Itô's lemma introduces a drift correction \( -\tfrac{1}{2}\sigma_j^2 \).</li>
 *   <li>The drift weight factor \( g(L) = L \) (extra \( L_l \) factor in the measure sum,
 *       arising from the change of variable \( \mathrm{d}\log L = \mathrm{d}L / L \)).</li>
 * </ul>
 *
 * @author Felipe, GM-1, GM-2
 * @see StateSpaceTransform
 * @see NormalStateSpaceTransform
 */
public class LogNormalStateSpaceTransform implements StateSpaceTransform {

	@Override
	public double getInitialState(final double rate) {
		return Math.log(Math.max(rate, 0));
	}

	@Override
	public RandomVariable applyTransform(final RandomVariable internalState) {
		return internalState.exp();
	}

	@Override
	public RandomVariable applyInverseTransform(final RandomVariable rate) {
		return rate.log();
	}

	@Override
	public RandomVariable getDriftWeightFactor(final RandomVariable liborRate) {
		return liborRate;
	}

	@Override
	public RandomVariable getItoCorrection(final RandomVariable variance) {
		return variance.mult(-0.5);
	}
}