package edu.jhu.Barbara.cs335.hw5.algorithm;

import java.io.Serializable;

import edu.jhu.Barbara.cs335.hw5.data.State;

/**
 * Implementers of this interface provide a means by which rewards can be assigned for certain states.
 * @author Zachary Palmer
 */
public interface RewardFunction extends Serializable
{
	/**
	 * Provides the reward for a given state.
	 * @param state The state to reward.
	 */
	public double reward(State state);
}
