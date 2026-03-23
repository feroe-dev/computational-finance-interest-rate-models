package net.finmath.lecture.computationalfinance.project.statespace;

import net.finmath.stochastic.RandomVariable;
import net.finmath.stochastic.Scalar;

/**
 * Normal (Bachelier) state-space transform for the LIBOR Market Model.
 *
 * <p>
 * Under the normal state space, each forward rate \( L_j \) is simulated directly
 * without any transformation:
 * \[
 *   X_j = L_j, \qquad f(X) = X \text{ (identity)}.
 * \]
 * Consequently:
 * <ul>
 *   <li>The initial state equals the market rate: \( X_0 = L_0 \).</li>
 *   <li>The state-space transform and its inverse are both the identity.</li>
 *   <li>No Itô correction is needed in the drift.</li>
 *   <li>The drift weight factor \( g(L) = 1 \) (no extra \( L \) factor in the measure sum).</li>
 * </ul>
 *
 * @author Felipe, GM-1, GM-2
 * @see StateSpaceTransform
 * @see LogNormalStateSpaceTransform
 */
public class NormalStateSpaceTransform implements StateSpaceTransform {

	@Override
	public double getInitialState(final double rate, final int liborIndex) {
		return rate;
	}

	@Override
	public RandomVariable applyTransform(final RandomVariable internalState, final int liborIndex) {
		return internalState;
	}

	@Override
	public RandomVariable applyInverseTransform(final RandomVariable rate, final int liborIndex) {
		return rate;
	}

	@Override
	public RandomVariable getDriftWeightFactor(final RandomVariable liborRate, final int liborIndex) {
		return Scalar.of(1.0);
	}

	@Override
	public RandomVariable getItoCorrection(final RandomVariable variance, final RandomVariable liborRate, final int liborIndex) {
		return Scalar.of(0.0);
	}
}