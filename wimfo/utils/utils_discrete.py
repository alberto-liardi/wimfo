import numpy as np

def test_pmf():
    """
    Returns a dictionary of predefined joint probability mass functions (PMFs)
    for four binary random variables (each taking values in {0, 1}).

    This function includes the following distributions:
    - "Independent"
    - "COPY_transfer"
    - "COPY"
    - "XOR"
    - "down_XOR"
    - "up_XOR"
    - "transfer"

    Returns:
        dict: A dictionary mapping scenario names (str) to 4D NumPy arrays 
              of shape (2, 2, 2, 2), representing the joint PMFs over 
              binary variables (X1, X2, X3, X4).
    """

    noise = 2**(-32)

    p = 1/8
    xor = {
    (0, 0, 0, 0): p
    , (0, 0, 0, 1): noise
    , (0, 0, 1, 0): noise
    , (0, 0, 1, 1): p
    , (0, 1, 0, 0): noise
    , (0, 1, 0, 1): p
    , (0, 1, 1, 0): p
    , (0, 1, 1, 1): noise 
    , (1, 0, 0, 0): noise 
    , (1, 0, 0, 1): p
    , (1, 0, 1, 0): p
    , (1, 0, 1, 1): noise 
    , (1, 1, 0, 0): p
    , (1, 1, 0, 1): noise 
    , (1, 1, 1, 0): noise 
    , (1, 1, 1, 1): p
    }

    copy_transfer = {
    (0, 0, 0, 0): p
    , (0, 0, 0, 1): noise
    , (0, 0, 1, 0): p
    , (0, 0, 1, 1): noise
    , (0, 1, 0, 0): p
    , (0, 1, 0, 1): noise
    , (0, 1, 1, 0): p
    , (0, 1, 1, 1): noise 
    , (1, 0, 0, 0): noise 
    , (1, 0, 0, 1): p
    , (1, 0, 1, 0): noise
    , (1, 0, 1, 1): p 
    , (1, 1, 0, 0): noise
    , (1, 1, 0, 1): p 
    , (1, 1, 1, 0): noise 
    , (1, 1, 1, 1): p
    }

    p = 1/4
    transfer = {
    (0, 0, 0, 0): p
    , (0, 0, 0, 1): noise
    , (0, 0, 1, 0): noise
    , (0, 0, 1, 1): noise
    , (0, 1, 0, 0): noise
    , (0, 1, 0, 1): noise
    , (0, 1, 1, 0): p
    , (0, 1, 1, 1): noise 
    , (1, 0, 0, 0): noise 
    , (1, 0, 0, 1): p
    , (1, 0, 1, 0): noise
    , (1, 0, 1, 1): noise 
    , (1, 1, 0, 0): noise
    , (1, 1, 0, 1): noise 
    , (1, 1, 1, 0): noise 
    , (1, 1, 1, 1): p
    }

    p = 1/4
    copy = {
    (0, 0, 0, 0): p
    , (0, 0, 0, 1): noise
    , (0, 0, 1, 0): noise
    , (0, 0, 1, 1): noise
    , (0, 1, 0, 0): noise
    , (0, 1, 0, 1): p
    , (0, 1, 1, 0): noise
    , (0, 1, 1, 1): noise 
    , (1, 0, 0, 0): noise 
    , (1, 0, 0, 1): noise
    , (1, 0, 1, 0): p
    , (1, 0, 1, 1): noise 
    , (1, 1, 0, 0): noise
    , (1, 1, 0, 1): noise 
    , (1, 1, 1, 0): noise 
    , (1, 1, 1, 1): p
    }

    p = 1/8
    down_XOR = {
    (0, 0, 0, 0): p
    , (0, 0, 0, 1): p
    , (0, 0, 1, 0): noise
    , (0, 0, 1, 1): noise
    , (0, 1, 0, 0): noise
    , (0, 1, 0, 1): noise
    , (0, 1, 1, 0): p
    , (0, 1, 1, 1): p 
    , (1, 0, 0, 0): noise 
    , (1, 0, 0, 1): noise
    , (1, 0, 1, 0): p
    , (1, 0, 1, 1): p 
    , (1, 1, 0, 0): p
    , (1, 1, 0, 1): p 
    , (1, 1, 1, 0): noise 
    , (1, 1, 1, 1): noise
    }

    up_XOR = {
    (0, 0, 0, 0): p
    , (0, 0, 0, 1): noise
    , (0, 0, 1, 0): noise
    , (0, 0, 1, 1): p
    , (0, 1, 0, 0): noise
    , (0, 1, 0, 1): p
    , (0, 1, 1, 0): p
    , (0, 1, 1, 1): noise 
    , (1, 0, 0, 0): p
    , (1, 0, 0, 1): noise
    , (1, 0, 1, 0): noise
    , (1, 0, 1, 1): p
    , (1, 1, 0, 0): noise
    , (1, 1, 0, 1): p
    , (1, 1, 1, 0): p
    , (1, 1, 1, 1): noise 
    }

    probs = {"Independent": np.repeat(1/16, 16).reshape(2,2,2,2)
    , "COPY_transfer": np.array(list(copy_transfer.values())).reshape(2,2,2,2)
    , "COPY": np.array(list(copy.values())).reshape(2,2,2,2)
    , "XOR": np.array(list(xor.values())).reshape(2,2,2,2)
    , "down_XOR": np.array(list(down_XOR.values())).reshape(2,2,2,2)
    , "up_XOR": np.array(list(up_XOR.values())).reshape(2,2,2,2)
    , "transfer": np.array(list(transfer.values())).reshape(2,2,2,2)
    }

    return probs

def estimate_discrete_distribution(x1, x2, y1, y2):
    """
    Estimate the joint probability distribution of 4 binary time series.
    
    Parameters:
    - x1, x2, y1, y2: Lists or 1D numpy arrays of the same length, containing binary values (0 or 1).
    
    Returns:
    - A 2x2x2x2 numpy array representing the joint probability distribution.
    """
    # Ensure the inputs are numpy arrays
    x1, x2, y1, y2 = map(np.asarray, (x1, x2, y1, y2))
    
    # Check that all series have the same length
    if not (len(x1) == len(x2) == len(y1) == len(y2)):
        raise ValueError("All input time series must have the same length.")
    
    # Initialize a 2x2x2x2 matrix to store joint counts
    joint_counts = np.zeros((2, 2, 2, 2))
    
    # Count occurrences of each combination
    for i in range(len(x1)):
        joint_counts[x1[i], x2[i], y1[i], y2[i]] += 1
    
    # Normalize to obtain probabilities
    joint_probabilities = joint_counts / np.sum(joint_counts)
    
    return joint_probabilities

def discrete_MI(P):
    """
    Calculate the mutual information I(X;Y) from a joint distribution P(X1, X2, Y1, Y2).

    Parameters:
    - P: ndarray of shape (2, 2, 2, 2), joint probability distribution.

    Returns:
    - Mutual information (in bits) as a float.
    """
    if P.ndim == 1:
        P = P.reshape((2, 2, 2, 2))

    P_X = np.sum(P, axis=(2, 3))  # Marginal over Y1, Y2
    P_Y = np.sum(P, axis=(0, 1))  # Marginal over X1, X2

    # Add small epsilon to avoid log(0)
    eps = 1e-10

    H_X = -np.sum(P_X * np.log2(P_X + eps))
    H_Y = -np.sum(P_Y * np.log2(P_Y + eps))
    H_XY = -np.sum(P * np.log2(P + eps))

    return H_X + H_Y - H_XY

if __name__ == '__main__':
    
   # Example: downward XOR
    x1 = np.random.randint(0, 2, 10000)
    x2 = np.random.randint(0, 2, 10000)
    y1 = np.random.randint(0, 2, 10000)
    y2 = (x1+x2)%2

    joint_distribution = estimate_discrete_distribution(x1, x2, y1, y2)
    print("Joint Probability Distribution:")
    print(joint_distribution)