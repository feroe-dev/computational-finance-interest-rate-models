package net.finmath.lecture.computationalfinance.project.statespace;

import net.finmath.stochastic.RandomVariable;

/**
 * Displaced log-normal state-space transform for the LIBOR Market Model.
 */
public class DisplacedLogNormalStateSpaceTransform implements StateSpaceTransform {

    private final double[] displacement;

    public DisplacedLogNormalStateSpaceTransform(double[] displacement) {
        this.displacement = displacement;
    }

    public DisplacedLogNormalStateSpaceTransform(double displacement, int numberOfLibors) {
        this.displacement = new double[numberOfLibors];
        for (int i = 0; i < numberOfLibors; i++) {
            this.displacement[i] = displacement;
        }
    }

    @Override
    public double getInitialState(double rate, int liborIndex) {
        return Math.log(Math.max(rate + displacement[liborIndex], 1e-10)); // Avoid log(0)
    }

    @Override
    public RandomVariable applyTransform(RandomVariable internalState, int liborIndex) {
        return internalState.exp().sub(displacement[liborIndex]);
    }
    
    @Override
    public RandomVariable applyInverseTransform(RandomVariable rate, int liborIndex) {
        return rate.add(displacement[liborIndex]).log();
    }

    @Override
    public RandomVariable getDriftWeightFactor(RandomVariable liborRate, int liborIndex) {
        return liborRate.add(displacement[liborIndex]);
    }

    @Override
    public RandomVariable getItoCorrection(RandomVariable variance, RandomVariable liborRate, int liborIndex) {
        return variance.mult(-0.5);
    }

}
