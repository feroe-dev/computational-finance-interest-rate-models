package net.finmath.lecture.computationalfinance.project.measure;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * This class is just an example for a unit test using Assertions.
 * Maybe you like to use unit tests.
 */
public class ExampleTest {

	@Test
	public void testExample() {

		double valueApproximated = 42.001;
		double valueExpeced = 42.0;

		double tolerance = 0.1;

		Assertions.assertEquals(valueExpeced, valueApproximated, tolerance, "Testing value");
	}
}
