import numpy as np
import math
import sys







###############################################################################
# Attitude Dynamics
###############################################################################


def euler_dynamics(w, I, L):
    '''
    This function computes the derivative of the angular velocity vector
    with respect to time given input angular velocity vector, moment of
    inertia matrix, and torque vector.  Angular velocity describes rate of
    change between body fixed B frame and inertial N frame.
    
    Parameters
    ------
    w : 3x1 numpy array, float
        angular velocity vector, coordinates in B frame
    I : 3x3 numpy array, float
        moment of inertia matrix, coordinates in B frame
    L : 3x1 numpy array, float
        torque vector, coordinates in B frame
    
    Returns
    ------
    w_dot : 3x1 numpy array, float
        time derivative of angular velocity vector
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 4.32
    '''

    wskew = skew_matrix(w)
    invI = np.linalg.inv(I)
    vect = np.dot(wskew, np.dot(I, w))

    w_dot = -np.dot(invI, vect) + np.dot(invI, L)
    
    return w_dot


def euler_dynamics_principal_axes(w, I, L):
    '''
    This function computes the derivative of the angular velocity vector
    with respect to time given input angular velocity vector, moment of
    inertia matrix, and torque vector.  Angular velocity describes rate of
    change between body fixed B frame and inertial N frame.
    
    Parameters
    ------
    w : 3x1 numpy array, float
        angular velocity vector, coordinates in B frame
    I : 3x3 numpy array, float
        moment of inertia matrix (diagonal), coordinates in B frame
    L : 3x1 numpy array, float
        torque vector, coordinates in B frame
    
    Returns
    ------
    w_dot : 3x1 numpy array, float
        time derivative of angular velocity vector
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 4.33
    '''
    
    w1 = float(w[0])
    w2 = float(w[1])
    w3 = float(w[2])
    
    I1 = float(I[0,0])
    I2 = float(I[1,1])
    I3 = float(I[2,2])
    
    L1 = float(L[0])
    L2 = float(L[1])
    L3 = float(L[2])
    
    dw1 = (1./I1)*((I2-I3)*w2*w3 + L1)
    dw2 = (1./I2)*((I3-I1)*w1*w3 + L2)
    dw3 = (1./I3)*((I1-I2)*w1*w2 + L3)
    
    w_dot = np.array([[dw1], [dw2], [dw3]])
    
    return w_dot




###############################################################################
# Attitude Kinematics
###############################################################################


def quat_derivative(q_BN, w_BN):
    '''
    This function computes the derivative of a quaternion with
    respect to time given an input angular velocity vector w.

    Parameters
    ------
    q_BN : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
        rotate vector coordinates from frame N to frame B
    w_BN : 3x1 numpy array, float
        angular velocity vector to specify rate of rotation between frame B 
        and N, vector coordinates provided in frame B

    Returns
    ------
    q_BN_dot : 3x1 numpy array, float
        time derivative of quaternion

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.104
    
    '''
    
    w1 = float(w_BN[0])
    w2 = float(w_BN[1])
    w3 = float(w_BN[2])    
    
    mat = np.array([[ 0.,  w3, -w2,  w1],
                    [-w3,  0.,  w1,  w2],
                    [ w2, -w1,  0.,  w3],
                    [-w1, -w2, -w3,  0.]])
    
    q_BN_dot = 0.5*np.dot(mat, q_BN)
    
    return q_BN_dot


def gibbs_derivative(g_BN, w_BN):
    '''
    This function computes the derivative of a Gibbs vector with
    respect to time given an input angular velocity vector w.

    Parameters
    ------
    g_BN : 3x1 numpy array, float
        Gibbs vector for frame rotation, rotate vector coordinates from 
        frame N to frame B
    w_BN : 3x1 numpy array, float
        angular velocity vector to specify rate of rotation between frame B 
        and N, vector coordinates provided in frame B

    Returns
    ------
    g_BN_dot : 3x1 numpy array, float
        time derivative of Gibbs vector

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.131    
    '''
    
    G = skew_matrix(g_BN)
    g_BN_dot = 0.5 * np.dot((np.eye(3) + G + np.dot(g_BN, g_BN.T)), w_BN)
    
    return g_BN_dot


def mrp_derivative(p_BN, w_BN):
    '''
    This function computes the derivative of a MRP vector with
    respect to time given an input angular velocity vector w.

    Parameters
    ------
    p_BN : 3x1 numpy array, float
        MRP vector for frame rotation, rotate vector coordinates from 
        frame N to frame B
    w_BN : 3x1 numpy array, float
        angular velocity vector to specify rate of rotation between frame B 
        and N, vector coordinates provided in frame B

    Returns
    ------
    p_BN_dot : 3x1 numpy array, float
        time derivative of MRP vector

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.155  
    '''
    
    P = skew_matrix(p_BN)
    pp = float(np.dot(p_BN.T, p_BN))
    p_BN_dot = 0.25 * np.dot(((1.-pp)*np.eye(3) + 2.*P + 
                               2.*np.dot(p_BN, p_BN.T)), w_BN)
    
    return p_BN_dot


def dcm_derivative(DCM_BN, w_BN):
    '''
    This function computes the derivative of a direction cosine matrix with
    respect to time given an input angular velocity vector w.

    Parameters
    ------
    DCM_BN : 3x3 numpy array, float
        direction cosine matrix to rotate vector coordinates from frame N to 
        frame B
    w_BN : 3x1 numpy array, float
        angular velocity vector to specify rate of rotation between frame B 
        and N, vector coordinates provided in frame B

    Returns
    ------
    DCM_BN_dot : 3x3 numpy array, float
        time derivative of direction cosine matrix

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.27-3.28
    
    '''
    
    wskew = skew_matrix(w_BN)
    DCM_BN_dot = np.dot(wskew, DCM_BN)
    
    return DCM_BN_dot
    

def euler321_derivative(theta1, theta2, theta3, wB):
    '''
    This function computes the derivative of a set of 3-2-1 Euler angles
    with respect to time given an input angular velocity vector w. 
    
    Angles used to transform vector coordinates from frame N to frame B.    
    For example, the 3-2-1 sequence of yaw-pitch-roll
    N -> b3(yaw) -> b2(pitch) -> b1(roll) -> B
    
    DCM_BN = DCM(roll) * DCM(pitch) * DCM(yaw)    
    rB = DCM_BN * rN
    wB given in frame B
    
    Parameters
    ------
    theta1 : float
        angle for first rotation, about axis 3
    theta2 : float
        angle for second rotation, about axis 2
    theta1 : float
        angle for third rotation, about axis 1        
    wB : 3x1 numpy array, float
        angular velocity vector to specify rate of rotation between frame N 
        and B, vector coordinates provided in frame B

    Returns
    ------
    dtheta1 : float
        angle rate about axis 3
    dtheta2 : float 
        angle rate about axis 2
    dtheta1 : float
        angle rate about axis 1

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.57
    
    '''
    
    
    c2 = math.cos(theta2)
    s2 = math.sin(theta2)
    c3 = math.cos(theta3)
    s3 = math.sin(theta3)
    
    mat = (1./c2) * np.array([[0.,   s3,       c3],
                              [0.,  c3*c2, -s3*c2],
                              [c2,  s3*s2,  c3*s2]])
    
    deuler = np.dot(mat, wB)
    dtheta1 = deuler[0]
    dtheta2 = deuler[1]
    dtheta3 = deuler[2]
    
    return dtheta1, dtheta2, dtheta3


def euler313_derivative(theta1, theta2, theta3, wB):
    '''
    This function computes the derivative of a set of 3-1-3 Euler angles
    with respect to time given an input angular velocity vector w.
    
    Angles used to transform vector coordinates from frame N to frame B.    
    For example, the 3-1-3 sequence of RAAN-Inc-ArgP
    N -> b3(RAAN) -> b1(Inc) -> b3(ArgP) -> B
    
    DCM_BN = DCM(ArgP) * DCM(Inc) * DCM(RAAN)    
    rB = DCM_BN * rN
    wB given in frame B
    
    Parameters
    ------
    theta1 : float
        angle for first rotation, about axis 3
    theta2 : float
        angle for second rotation, about axis 1
    theta1 : float
        angle for third rotation, about axis 3
    wB : 3x1 numpy array, float
        angular velocity vector to specify rate of rotation between frame N 
        and B, vector coordinates provided in frame B

    Returns
    ------
    dtheta1 : float
        angle rate about axis 3
    dtheta2 : float 
        angle rate about axis 1
    dtheta1 : float
        angle rate about axis 3

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.58
    
    '''
    
    
    c2 = math.cos(theta2)
    s2 = math.sin(theta2)
    c3 = math.cos(theta3)
    s3 = math.sin(theta3)
    
    mat = (1./c2) * np.array([[s3*s2,  c3,  0.],
                              [c3*s2, -s3,  0.],
                              [c2,     0.,  1.]])
    
    deuler = np.dot(mat, wB)
    dtheta1 = deuler[0]
    dtheta2 = deuler[1]
    dtheta3 = deuler[2]
    
    return dtheta1, dtheta2, dtheta3


###############################################################################
# Compositions
###############################################################################


def quat_composition(q_AB, q_BC):
    '''
    This function computes the composition of two quaternions to rotate from 
    frame B to frame A, and from frame C to frame B.  The output quaternion
    will rotate from frame C to frame A.
    
    va = DCM(q_AB)*vb
    vb = DCM(q_BC)*vc
    va = DCM(q_AC)*vc
    
    Parameters
    ------
    q_AB : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    q_BC : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)

    Returns
    ------
    q_AC : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)

    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.46
    
    '''
    
    q1 = float(q_AB[0])
    q2 = float(q_AB[1])
    q3 = float(q_AB[2])
    q4 = float(q_AB[3])
    
    mat = np.array([[ q4,  q3, -q2,  q1],
                    [-q3,  q4,  q1,  q2],
                    [ q2, -q1,  q4,  q3],
                    [-q1, -q2, -q3,  q4]])
    
    q_AC = np.dot(mat, q_BC)
    
    return q_AC


def gibbs_composition(g_AB, g_BC):
    '''
    This function computes the composition of two Gibbs vectors
    to rotate from frame B to frame A, and from frame C to frame B.  The 
    output Gibbs vector will rotate from frame C to frame A.
    
    va = DCM(g_AB)*vb
    vb = DCM(g_BC)*vc
    va = DCM(g_AC)*vc
    
    Parameters
    ------
    g_AB : 3x1 numpy array, float
        Gibbs vector to rotate vector coordinates from frame B to frame A
    g_BC : 3x1 numpy array, float
        Gibbs vector to rotate vector coordinates from frame C to frame B

    Returns
    ------
    g_AC : 3x1 numpy array, float
        Gibbs vector to rotate vector coordinates from frame C to
        frame A

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.126
    '''
    
    g_AC = (g_AB + g_BC - np.cross(g_AB, g_BC, axis=0))/(1. - np.dot(g_AB.T, g_BC))
    
    return g_AC


def mrp_composition(p_AB, p_BC):
    '''
    This function computes the composition of two MRP vectors
    to rotate from frame B to frame A, and from frame C to frame B.  The 
    output MRP vector will rotate from frame C to frame A.
    
    va = DCM(p_AB)*vb
    vb = DCM(p_BC)*vc
    va = DCM(p_AC)*vc
    
    Parameters
    ------
    p_AB : 3x1 numpy array, float
        MRP vector to rotate vector coordinates from frame B to frame A
    p_BC : 3x1 numpy array, float
        MRP vector to rotate vector coordinates from frame C to frame B

    Returns
    ------
    p_AC : 3x1 numpy array, float
        MRP vector to rotate vector coordinates from frame C to
        frame A

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.152
    '''
    
    
    p_AB2 = float(np.dot(p_AB.T, p_AB))
    p_BC2 = float(np.dot(p_BC.T, p_BC))
    pcross = np.cross(p_AB, p_BC, axis=0)
    pdot = float(np.dot(p_BC.T, p_AB))
    
    p_AC = ((1. - p_BC2)*p_AB + (1. - p_AB2)*p_BC - 2.*pcross) / \
        (1. + p_BC2*p_AB2 - 2.*pdot)
    
    return p_AC


def dcm_composition(DCM_AB, DCM_BC):
    '''
    This function computes the composition of two direction cosine matrices
    to rotate from frame B to frame A, and from frame C to frame B.  The 
    output DCM will rotate from frame C to frame A.
    
    va = DCM_AB*vb
    vb = DCM_BC*vc
    va = DCM_AC*vc
    
    Parameters
    ------
    DCM_AB : 3x3 numpy array, float
        direction cosine matrix to rotate vector coordinates from frame B to
        frame A
    DCM_BC : 3x3 numpy array, float
        direction cosine matrix to rotate vector coordinates from frame C to
        frame B

    Returns
    ------
    DCM_AC : 3x3 numpy array, float
        direction cosine matrix to rotate vector coordinates from frame C to
        frame A

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.20
    
    '''
    
    DCM_AC = np.dot(DCM_AB, DCM_BC)
    
    return DCM_AC
    

def dcm_principal_axis(axis, angle):
    '''
    This function returns the direction cosine matrix for a principal axis 
    rotation.
    
    Parameters
    ------
    axis : int
        principal axis for frame rotation [1, 2, or 3]
    angle : float
        angle of rotation [rad]
    
    Returns
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix for specified principal axis/angle rotation
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.32   
        
    '''
    
    c = math.cos(angle)
    s = math.sin(angle)
    
    if axis == 1:
        DCM = np.array([[1.,  0., 0.],
                        [0.,  c,  s],
                        [0., -s,  c]])
    elif axis == 2:
        DCM = np.array([[c,  0., -s],
                        [0., 1.,  0.],
                        [s,  0.,  c]])
    elif axis == 3:
        DCM = np.array([[ c,  s,  0.],
                        [-s,  c,  0.],
                        [ 0., 0., 1.]])
    else:
        sys.stderr('WARNING: Invalid axis parameter!!')
    
    
    return DCM


def euler_angles(sequence, theta1, theta2, theta3):
    '''
    This function computes the direction cosine matrix corresponding to a 
    sequence of 3 Euler angle rotations about principal axes. This is
    generally a set of 3 rotations about body fixed axes to transform 
    vector coordinates from an inertial to body fixed frame.
    
    For example, the 3-2-1 sequence of yaw-pitch-roll
    N -> b3(yaw) -> b2(pitch) -> b1(roll) -> B
    
    The output DCM_BN is such that
    DCM_BN = DCM(roll) * DCM(pitch) * DCM(yaw)
    
    rB = DCM_BN * rN
    
    Parameters
    ------
    sequence : 3 element list, int
        principal axes for frame rotation in order from frame N to B 
        [1, 2, or 3]
    theta1 : float
        angle of first rotation [rad]
    theta2 : float
        angle of second rotation [rad]
    theta3 : float
        angle of third rotation [rad]
    
    Returns
    ------
    DCM_euler : 3x3 numpy array, float
        direction cosine matrix for full Euler sequence from frame N to B
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.33
    
    '''
    
    # Compute individual axis rotation DCMs
    R1 = dcm_principal_axis(sequence[0], theta1)
    R2 = dcm_principal_axis(sequence[1], theta2)
    R3 = dcm_principal_axis(sequence[2], theta3)
    
    # Compute composition for full sequence
    DCM_euler = np.dot(R3, np.dot(R2, R1))
    
    return DCM_euler



###############################################################################
# Frame Rotations
###############################################################################


def quat_rotate(q_BA, va):
    '''
    This function rotates the coordinates of a vector from frame A to frame 
    B per the given quaternion.
    
    vb = DCM(q_BA)*va
    
    Parameters
    ------
    q_BA : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    va : 3x1 numpy array, float
        coordinate vector in frame A

    Returns
    ------
    vb : 3x1 numpy array, float
        coordinate vector in frame B    
    '''
    
    # Actual formula is vb = q*va*qinv
    vaq = np.append(va, 0.)
    qinv = quat_inverse(q_BA)    
    vb = quat_composition(q_BA, quat_composition(vaq, qinv))[0:3]
    
    # Other option to just compute a DCM and mulitply    
#    DCM_BA = quat2dcm(q_BA)
#    vb = dcm_rotate(DCM_BA, va)

    return vb


def dcm_rotate(DCM_BA, va):
    '''
    This function rotates the coordinates of a vector from frame A to frame 
    B per the given direction cosine matrix (DCM).
    
    vb = DCM_BA*va
    
    Parameters
    ------
    DCM_BA : 3x3 numpy array, float
        direction cosine matrix
    va : 3x1 numpy array, float
        coordinate vector in frame A

    Returns
    ------
    vb : 3x1 numpy array, float
        coordinate vector in frame B

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.17
    
    '''
    
    vb = np.dot(DCM_BA, va)
    
    return vb




###############################################################################
# General Math
###############################################################################


def quat_inverse(q):
    '''
    This function computes the inverse quaternion, to perform the reverse
    frame rotation.
    
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Returns
    ------
    qinv : 4x1 numpy array, float
        quaternion for reverese frame rotation (vector part on top, q4 is 
        scalar)
    '''
    
    qinv = -q
    qinv[3] = q[3]
    
    return qinv


def quat_scalar2first(q):
    '''
    This function switches the order of quaternion components to move the 
    scalar part from the last element to the first element.
    
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Returns
    ------
    qflip : 4x1 numpy array, float
        quaternion for frame rotation (vector part on bottom, q1 is scalar)
    '''
    
    qflip = np.zeros(q.shape)
    qflip[0] = q[3]
    qflip[1:4] = q[0:3]
    
    return qflip


def quat_scalar2last(q):
    '''
    This function switches the order of quaternion components to move the 
    scalar part from the first element to the last element.
    
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on bottom, q1 is scalar)
    
    Returns
    ------
    qflip : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    '''
    
    qflip = np.zeros(q.shape)
    qflip[3] = q[0]
    qflip[0:3] = q[1:4]
    
    return qflip

 
def gibbs_inverse(g):
    '''
    This function computes the inverse Gibbs vector, to perform the reverse
    frame rotation.
    
    Parameters
    ------
    g : 3x1 numpy array, float
        Gibbs vector for frame rotation
    
    Returns
    ------
    ginv : 3x1 numpy array, float
        Gibbs vector for reverese frame rotation
    '''
    
    ginv = -g
    
    return ginv


def mrp_inverse(p):
    '''
    This function computes the inverse MRP vector, to perform the reverse
    frame rotation.
    
    Parameters
    ------
    p : 3x1 numpy array, float
        MRP vector for frame rotation
    
    Returns
    ------
    ginv : 3x1 numpy array, float
        MRP vector for reverese frame rotation
        
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.150
    '''
    
    pinv = -p
    
    return pinv


def mrp_shadow(p):
    '''
    This function compute the shadow modified Rodrigues parameter vector given
    an initial MRP vector.  The shadow set is used to avoid singularities.
    Regular MRPs are singular at +/-360 deg, shadow MRPs are singular at 0 deg.
    Generally switch between sets at +/-180 deg, the regular + shadow MRPs
    provide a minimal attitude representation that is also nonsingular.
    
    Parameters
    ------
    p : 3x1 numpy array, float
        MRP vector
    
    Returns
    ------
    ps : 3x1 numpy array, float
        shadow MRP vector
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.142
    
    '''
    
    # At 180 deg, pp = 1, ps = -p
    pp = float(np.dot(p.T, p))   
    ps = -(1./pp) * p
    
    return ps


def skew_matrix(w):
    '''
    This function returns the skew symmetric matrix from an input vector, such
    that the cross product is equal to the matrix multiplication.
    
    cross(w,b) = wskew*b
    
    Parameters
    ------
    w : 3x1 numpy array, float
        coordinate vector

    Returns
    ------
    wskew : 3x3 numpy array, float
        skew symmetric matrix to perform cross product

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.23

    '''
    
    wskew = np.array(([[   0, -w[2], w[1]],
                       [ w[2],   0, -w[0]],
                       [-w[1], w[0],   0]]))
    
    return wskew



###############################################################################
# Conversion between Attitude Representations
###############################################################################


def quat2ehatphi(q):
    '''
    This function returns a principal rotation vector and angle given an 
    input quaternion.
       
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Returns
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.33
    '''
    
    qvec = q[0:3]
    q4 = float(q[3])    
    
    phi = 2.*math.acos(q4)
    
    if math.sin(phi/2.) != 0.:
        ehat = (1./math.sin(phi/2.))*qvec
    else:
        # phi = 0. is only condition to get here, just choose rotation axis
        ehat = np.array([[0.], [0.], [1.]])
        
    
    return ehat, phi
    
    
def ehatphi2quat(ehat, phi):
    '''
    This function returns a quaternion given an input principal rotation 
    vector and angle.
       
    Parameters
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Returns
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.33
    ''' 
    
    q = np.zeros((4,1))
    q[0:3] = math.sin(phi/2.)*ehat
    q[3] = math.cos(phi/2.)
    
    return q
  
    
def quat2mrp(q):
    '''
    This function returns a modified Rodrigues parameter vector given an 
    input quaternion.
       
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Returns
    ------
    p : 3x1 numpy array, float
        MRP vector for frame rotation
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.137
    '''
    
    q1 = float(q[0])
    q2 = float(q[1])
    q3 = float(q[2])
    q4 = float(q[3])
    
    p = np.array([[q1/(1.+q4)], [q2/(1.+q4)], [q3/(1.+q4)]])
    
    return p
    
    
def mrp2quat(p):
    '''
    This function returns a quaternion given an input modified Rodrigues
    parameter vector.
       
    Parameters
    ------
    p : 3x1 numpy array, float
        MRP vector for frame rotation        
    
    Returns
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.138
    '''
    
    p1 = float(p[0])
    p2 = float(p[1])
    p3 = float(p[2])
    pp = float(np.dot(p.T, p))

    q = np.array([[2.*p1/(1.+pp)],
                  [2.*p2/(1.+pp)],
                  [2.*p3/(1.+pp)],
                  [(1.-pp)/(1.+pp)]])    
    
    return q


def quat2grp(q, a, f=None):
    '''
    This function returns a generalized Rodrigues parameter vector given an 
    input quaternion.
    
    Gibbs vector - (a = 0, f = 1)
    MRP vector - (a = 1, f = 1)
    Angle of rotation (small angles) - (f = 2(a+1))
    
    This last choice can be used with error GRP to get error roll-pitch-yaw.
       
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    a : float
        scaling parameter between 0 and 1
    f : float (optional)
        scaling parameter, default = None will cause f = 2(a+1)
        
    
    Returns
    ------
    p : 3x1 numpy array, float
        GRP vector for frame rotation
    
    Reference
    ------
    Crassidis and Markley, "Unscented Filtering for Spacecraft Attitude 
        Estimation," 2003.
    '''
    
    if not f:
        f = 2.*(a + 1.)
    
    qvec = q[0:3].reshape(3,1)
    
    p = (f / (a + float(q[3]))) * qvec
    
    return p
    
    
def grp2quat(p, a, f=None):
    '''
    This function returns a quaternion given an input generalized Rodrigues
    parameter vector.
    
    Gibbs vector - (a = 0, f = 1)
    MRP vector - (a = 1, f = 1)
    Angle of rotation (small angles) - (f = 2(a+1))
    
    This last choice can be used with error GRP to get error roll-pitch-yaw.
       
    Parameters
    ------
    p : 3x1 numpy array, float
        GRP vector for frame rotation 
    a : float
        scaling parameter between 0 and 1
    f : float (optional)
        scaling parameter, default = None will cause f = 2(a+1)
    
    Returns
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Reference
    ------
    Crassidis and Markley, "Unscented Filtering for Spacecraft Attitude 
        Estimation," 2003.
    '''
    
    if not f:
        f = 2.*(a + 1.)
        
    pnorm2 = float(np.dot(p.T, p))
    q4 = (-a*pnorm2 + f*np.sqrt(f**2. + (1. - a**2.)*pnorm2))/(f**2. + pnorm2)
    qvec = ((a + q4)/f) * p
    
    q = np.append(qvec, q4).reshape(4,1)
    
    return q


def quat2gibbs(q):
    '''
    This function returns a Gibbs vector given an input quaternion.
       
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Returns
    ------
    g : 3x1 numpy array, float
        Gibbs vector for frame rotation
    
    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.48
    '''
    
    q1 = float(q[0])
    q2 = float(q[1])
    q3 = float(q[2])
    q4 = float(q[3])
    
    if q4 == 0:
        sys.stderr.write('WARNING: 180 degree rotation, Gibbs vector undefined!!')
        g = float('nan') * np.ones((3,1))
    else:    
        g = np.array([[q1/q4], [q2/q4], [q3/q4]])
    
    return g
    

def gibbs2quat(g):
    '''
    This function returns a quaternion given an input Gibbs vector.
       
    Parameters
    ------
    g : 3x1 numpy array, float
        Gibbs vector for frame rotation
    
    Returns
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.115
    '''
    
    g1 = float(g[0])
    g2 = float(g[1])
    g3 = float(g[2])
    gg = float(np.dot(g.T, g))
    denom = np.sqrt(1. + gg)
    
    q = np.array([[g1/denom], [g2/denom], [g3/denom], [1./denom]])
    
    return q
    

def quat2dcm(q):
    '''
    This function returns a direction cosine matrix given an input quaternion.
       
    Parameters
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Returns
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix
    
    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.38
    '''
    
    qvec = q[0:3]
    q4 = float(q[3])
    qskew = skew_matrix(qvec)

    DCM = (q4**2. - np.dot(qvec.T, qvec))*np.eye(3) + 2.*np.dot(qvec, qvec.T) \
        - 2.*q4*qskew

    return DCM
    

def dcm2quat(DCM):
    '''
    This function returns a direction cosine matrix given an input quaternion.
       
    Parameters
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix
    
    Returns
    ------
    q : 4x1 numpy array, float
        quaternion for frame rotation (vector part on top, q4 is scalar)
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.95-3.96
    '''

    # Sheppard Method to avoid divide by zero
    q1_2 = 0.25*(1. + 2.*DCM[0,0] - np.trace(DCM))
    q2_2 = 0.25*(1. + 2.*DCM[1,1] - np.trace(DCM))
    q3_2 = 0.25*(1. + 2.*DCM[2,2] - np.trace(DCM))
    q4_2 = 0.25*(1. + np.trace(DCM))
    
    ind = np.argmax([q1_2, q2_2, q3_2, q4_2])
    
    if ind == 0:
        q1 = np.sqrt(q1_2)
        q2 = (1./(4.*q1))*(DCM[0,1] + DCM[1,0])
        q3 = (1./(4.*q1))*(DCM[2,0] + DCM[0,2])
        q4 = (1./(4.*q1))*(DCM[1,2] - DCM[2,1])
    elif ind == 1:
        q2 = np.sqrt(q2_2)
        q1 = (1./(4.*q2))*(DCM[0,1] + DCM[1,0])
        q3 = (1./(4.*q2))*(DCM[1,2] + DCM[2,1])
        q4 = (1./(4.*q2))*(DCM[2,0] - DCM[0,2])
    elif ind == 2:
        q3 = np.sqrt(q3_2)
        q1 = (1./(4.*q3))*(DCM[2,0] + DCM[0,2])
        q2 = (1./(4.*q3))*(DCM[1,2] + DCM[2,1])
        q4 = (1./(4.*q3))*(DCM[0,1] - DCM[1,0])
    elif ind == 3:
        q4 = np.sqrt(q4_2)
        q1 = (1./(4.*q4))*(DCM[1,2] - DCM[2,1])
        q2 = (1./(4.*q4))*(DCM[2,0] - DCM[0,2])
        q3 = (1./(4.*q4))*(DCM[0,1] - DCM[1,0])
        
    
    q = np.array([[q1], [q2], [q3], [q4]])       
    
    return q


def mrp2gibbs(p):
    '''
    This function returns a Gibbs vector given an input modified Rodrigues 
    parameters vector.
    
    Parameters
    ------
    p : 3x1 numpy array, float
        MRP vector
    
    Returns
    ------
    g : 3x1 numpy array, float
        Gibbs vector
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.140
    '''
    
    pp = float(np.dot(p.T, p))
    g = (2./(1. - pp)) * p
    
    return g


def gibbs2mrp(g):
    '''
    This function returns a modified Rodrigues parameter vector given an input
    Gibbs vector.
    
    Parameters
    ------
    g : 3x1 numpy array, float
        Gibbs vector
    
    Returns
    ------
    p : 3x1 numpy array, float
        MRP vector
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.141
    '''
    
    p = (1./(1. + np.sqrt(1. + np.dot(g.T, g)))) * g
    
    return p


def mrp2ehatphi(p):
    '''
    This function returns a principal rotation vector and angle given an input 
    modified Rodrigues parameters vector.
    
    Parameters
    ------
    p : 3x1 numpy array, float
        MRP vector
    
    Returns
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.139
    '''
    
    phi = 4.*math.atan(np.linalg.norm(p))
    ehat = 1./(math.tan(phi/4.)) * p
    
    return ehat, phi


def ehatphi2mrp(ehat, phi):
    '''
    This function returns a modified Rodrigues parameters vector given input 
    principal rotation vector and angle.
    
    Parameters
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Returns
    ------
    p : 3x1 numpy array, float
        MRP vector    
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.139
    '''
    
    p = ehat*math.tan(phi/4.)

    return p


def mrp2dcm(p):
    '''
    This function returns a direction cosine matrix given input modified
    Rodrigues parameter vector.
    
    Parameters
    ------
    p : 3x1 numpy array, float
        MRP vector
    
    Returns
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix

    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.147
    '''
    
    P = skew_matrix(p)
    pp = float(np.dot(p.T, p))
    DCM = np.eye(3) + (1./((1.+pp)**2.))*(8.*np.dot(P, P) - 4.*(1.-pp)*P)
    
    return DCM
    

def dcm2mrp(DCM):
    '''
    This function returns a modified Rodrigues parameter vector given input 
    direction cosine matrix.
    
    Parameters
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix
    
    Returns
    ------
    p : 3x1 numpy array, float
        MRP vector    
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.149
    '''
    
    if np.trace(DCM) == -1.:
        sys.stderr.write('WARNING: 180 degree rotation, transforming to quaternion')
        q = dcm2quat(DCM)
        p = quat2mrp(q)
    else:
        zeta = np.sqrt(1. + np.trace(DCM))
        p = 1./(zeta*(zeta+2.)) * np.array([[DCM[1,2] - DCM[2,1]],
                                            [DCM[2,0] - DCM[0,2]],
                                            [DCM[0,1] - DCM[1,0]]])

    return p


def gibbs2dcm(g):
    '''
    This function returns a direction cosine matrix given input Gibbs vector.
    
    Parameters
    ------
    g : 3x1 numpy array, float
        Gibbs vector
    
    Returns
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix

    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.50
    '''
    
    G = skew_matrix(g)
    gg = float(np.dot(g.T, g))
    
    DCM = 1./(1. + gg) * ((1 - gg)*np.eye(3) + 2.*np.dot(g, g.T) - 2.*G)
    
    return DCM
    
    
def dcm2gibbs(DCM):
    '''
    This function returns a Gibbs vector given input direction cosine matrix.
    
    Parameters
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix
    
    Returns
    ------
    g : 3x1 numpy array, float
        Gibbs vector    
    
    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.51
    '''
    
    if np.trace(DCM) == -1.:
        sys.stderr.write('WARNING: 180 degree rotation, Gibbs vector undefined!!')
        g = float('nan') * np.ones((3,1))
    else:
        g = 1./(1. + np.trace(DCM)) * np.array([[DCM[1,2] - DCM[2,1]],
                                                [DCM[2,0] - DCM[0,2]],
                                                [DCM[0,1] - DCM[1,0]]])
    
    return g


def gibbs2ehatphi(g):
    '''
    This function returns a principal rotation vector and angle given an input 
    Gibbs vector.
    
    Parameters
    ------
    g : 3x1 numpy array, float
        Gibbs vector
    
    Returns
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.48
    '''  
    
    phi = 2.*math.atan(np.linalg.norm(g))
    ehat = 1./(math.tan(phi/2)) * g
    
    return ehat, phi
    
    
def ehatphi2gibbs(ehat, phi):
    '''
    This function returns a Gibbs vector given input principal rotation vector
    and angle.
    
    Parameters
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Returns
    ------
    g : 3x1 numpy array, float
        Gibbs vector    
    
    Reference
    ------
    B. Wie, "Space Vehicle Dynamics and Control," 2nd ed., 2008. Eq. 5.48
    '''
    
    g = ehat*math.tan(phi/2.)
    
    return g


def dcm2euler321(DCM):
    '''
    This function returns the Euler angles for a 3-2-1 sequence of principal
    axis rotations (e.g. yaw-pitch-roll).
    
    Note this parameterization has a singularity when theta2 = +/-90 deg
    
    Parameters
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix for full Euler rotation
    
    Returns
    ------
    theta1 : float
        angle for first rotation, about axis 3
    theta2 : float
        angle for second rotation, about axis 2
    theta1 : float
        angle for third rotation, about axis 1
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.35        
    
    '''
    
    theta1 = math.atan2(DCM[0,1], DCM[0,0])
    theta3 = math.atan2(DCM[1,2], DCM[2,2])
    
    try:
        theta2 = -math.asin(DCM[0,2])
    except:
        theta2 = -math.asin(round(DCM[0,2]))
    
    return theta1, theta2, theta3


def dcm2euler123(DCM):
    '''
    This function returns the Euler angles for a 1-2-3 sequence of principal
    axis rotations (e.g. roll-pitch-yaw).
    
    Note this parameterization has a singularity when theta2 = +/-90 deg
    
    Parameters
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix for full Euler rotation
    
    Returns
    ------
    theta1 : float
        angle for first rotation, about axis 1
    theta2 : float
        angle for second rotation, about axis 2
    theta1 : float
        angle for third rotation, about axis 3
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  No Equation, developed from Eq 3.35      
    
    '''
    
    theta1 = -math.atan2(DCM[2,1], DCM[2,2])
    theta3 = -math.atan2(DCM[1,0], DCM[0,0])
    try:
        theta2 = math.asin(DCM[2,0])
    except:
        theta2 = math.asin(round(DCM[2,0]))
    
    return theta1, theta2, theta3


def dcm2euler313(DCM):
    '''
    This function returns the Euler angles for a 3-1-3 sequence of principal
    axis rotations (e.g. RAAN-Inc-ArgP).
    
    Note this parameterization has a singularity when theta2 = 0 or 180 deg
    
    Parameters
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix for full Euler rotation
    
    Returns
    ------
    theta1 : float
        angle for first rotation, about axis 3
    theta2 : float
        angle for second rotation, about axis 1
    theta1 : float
        angle for third rotation, about axis 3
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.35        
    
    '''
    
    theta1 = math.atan2(DCM[2,0], -DCM[2,1])
    theta2 = math.acos(DCM[2,2])
    theta3 = math.atan2(DCM[0,2], DCM[1,2])
    
    return theta1, theta2, theta3


def dcm2ehatphi(DCM):
    '''
    This function returns a principal rotation vector and angle given an input 
    direction cosine matrix.
    
    Parameters
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix
    
    Returns
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.73-3.74
    '''    
    
    c = 0.5 * (DCM[0,0] + DCM[1,1] + DCM[2,2] - 1.)
    phi = math.acos(c)
    
    if math.sin(phi) != 0:
        ehat = (1./(2.*math.sin(phi))) * np.array([[DCM[1,2] - DCM[2,1]],
                                              [DCM[2,0] - DCM[0,2]],
                                              [DCM[0,1] - DCM[1,0]]])
    else:
        # phi = 0 or 180 degrees, quat2ehatphi will deal with issue
        q = dcm2quat(DCM)
        ehat, phi = quat2ehatphi(q)
    
    return ehat, phi
    
    
def ehatphi2dcm(ehat, phi):
    '''
    This function returns a direction cosine matrix given an input principal
    rotation vector and angle.
    
    Parameters
    ------
    ehat : 3x1 numpy array, float
        unit vector to specify axis of rotation (same in both frames)
    phi : float
        angle of rotation [rad]
    
    Returns
    ------
    DCM : 3x3 numpy array, float
        direction cosine matrix
    
    Reference
    ------
    H. Schaub and J.L. Junkins, "Analytical Mechanics of Space Systems,"
    2nd ed., 2009.  Eq. 3.72
    '''    
    
    c = math.cos(phi)
    s = math.sin(phi)
    sig = 1 - c
    e1 = float(ehat[0])
    e2 = float(ehat[1])
    e3 = float(ehat[2])
        
    
    DCM = np.array([[e1**2.*sig + c,    e1*e2*sig + e3*s,  e1*e3*sig - e2*s],
                    [e2*e1*sig - e3*s,   e2**2.*sig + c,   e2*e3*sig + e1*s],
                    [e3*e1*sig + e2*s,  e3*e2*sig - e1*s,    e3**2.*sig + c]])
    
    
    return DCM



###############################################################################
# Unit Test Functions
###############################################################################
#
#
#
#def test_skew_matrix():
#    
#    a = np.reshape([2., -3., 4.], (3,1))
#    b = np.reshape([5., 2., -6.], (3,1))
#    
#    c = np.cross(a,b,axis=0)
#    
#    d = np.dot(skew_matrix(a), b)
#    
##    print c
##    print d
#    
#    
#    
#    return
#    
#
    
#def test_euler():
#    
#    eulerB = [90., 0., -90.]
#    sequence = [3,2,1]
#    
#
#    
#    DCM_BN = euler_angles(sequence, eulerB[0]*pi/180., eulerB[1]*pi/180.,
#                          eulerB[2]*pi/180.)
#    
#    print(DCM_BN)
#    
#    yaw, pitch, roll = dcm2euler321(DCM_BN)
#    
#    print(roll*180/pi)
#    print(pitch*180/pi)
#    print(yaw*180/pi)
#    
#    
#    eulerB = [-90., -90., 0.]
#    sequence = [1,2,3]
#    
#    DCM_BN = euler_angles(sequence, eulerB[0]*pi/180., eulerB[1]*pi/180.,
#                          eulerB[2]*pi/180.)
#    
#    print(DCM_BN)
#    
#    roll, pitch, yaw = dcm2euler123(DCM_BN)
#    
#    print(roll*180/pi)
#    print(pitch*180/pi)
#    print(yaw*180/pi)
#    
#    
#    
#    return
    
#def test_att_error():
#    
#    
#    euler123 = [1., 10., 4.]
#    sequence = [1, 2, 3]
#    
#    DCM_BN = euler_angles(sequence, euler123[0]*pi/180., euler123[1]*pi/180.,
#                          euler123[2]*pi/180.)
#    
#    mrp_BN = dcm2mrp(DCM_BN)
#    gibbs_BN = dcm2gibbs(DCM_BN)  
#    
#    q_BN = dcm2quat(DCM_BN)
#    
#    grp_BN = quat2grp(q_BN, 1, 1)
#    
#    print('MRP', mrp_BN)
#    print('MRP_check', grp_BN)
#    
#    grp_BN = quat2grp(q_BN, 0, 1)
#    
#    print('Gibbs', gibbs_BN)
#    print('Gibbs check', grp_BN)
#    
#    grp_BN = quat2grp(q_BN, 1)
#    
#    print('Angles check')
#    print(mrp_BN*4*180/pi)
#    print(grp_BN*180/pi)
#    
#    q_check = grp2quat(grp_BN, 1)
#    
#    print('q', q_BN)
#    print('q check', q_check)
#    
#    
#    
#    return

#test_att_error()

#test_euler()
    
#def test_bug():
#    
#    q_IB = np.array([[ 0.46867662], [ 0.17383645], [-0.83205448], [-0.24043389]])
#    
#    n_hat = np.array([[-1.], [0.], [0.]])
#    
#    print(quat_rotate(q_IB, n_hat))
#    
#    
#    return
#
#test_bug()

#def test_parameter_conversions():
#    '''
#    Schaub and Junkins Example 3.2, 3.3
#    '''    
#    
#    
#    eulerB = [30., -45., 60.]
#    eulerF = [10., 25., -15.]
#    
#    sequence = [3,2,1]
#    
#    DCM_BN = euler_angles(sequence, eulerB[0]*pi/180., eulerB[1]*pi/180.,
#                          eulerB[2]*pi/180.)
#    
#    DCM_FN = euler_angles(sequence, eulerF[0]*pi/180., eulerF[1]*pi/180.,
#                          eulerF[2]*pi/180.)
#                          
#    DCM_BF = dcm_composition(DCM_BN, DCM_FN.T)
#    
#    yaw, pitch, roll = dcm2euler321(DCM_BF)
#    
#    ehat_FN, phi_FN = dcm2ehatphi(DCM_FN)
#    
#    DCM_FN = ehatphi2dcm(ehat_FN, phi_FN)
#    
##    print 'BN', DCM_BN
##    print
##    print 'FN', DCM_FN
##    print
##    print 'BF', DCM_BF
##    print
##    print 'yaw', yaw*180./pi
##    print 'pitch', pitch*180./pi
##    print 'roll', roll*180./pi
##    print
##    print 'ehat', ehat_FN
##    print 'phi', phi_FN*180./pi
#    
#    
#    quat_FN = dcm2quat(DCM_FN)
#    DCM_FN = quat2dcm(quat_FN)
#    ehat_FN, phi_FN = quat2ehatphi(quat_FN)
#    quat_FN2 = ehatphi2quat(ehat_FN, phi_FN)
#    
#    print
#    print 'FN', DCM_FN
#    print
#    print 'quat', quat_FN
#    print 'quat2', quat_FN2
#    print 'ehat', ehat_FN
#    print 'phi', phi_FN*180./pi
#    
#    gibbs_FN = quat2gibbs(quat_FN)
#    quat_FN = gibbs2quat(gibbs_FN)
#    ehat_FN, phi_FN = gibbs2ehatphi(gibbs_FN)
#    gibbs_FN2 = ehatphi2gibbs(ehat_FN, phi_FN)
#    
#    print
#    print 'gibbs_FN', gibbs_FN
#    print 'quat_FN', quat_FN
#    print 'ehat', ehat_FN
#    print 'phi', phi_FN*180./pi
#    print 'gibbs_FN2', gibbs_FN2
#    
#    DCM_FN = gibbs2dcm(gibbs_FN)
#    gibbs_FN = dcm2gibbs(DCM_FN)
#    
#    print
#    print 'gibbs_FN', gibbs_FN
#    print 'DCM_FN', DCM_FN
#    
#    print
#    print 'MRPs'
#    
#    p_FN = quat2mrp(quat_FN)
#    quat_FN = mrp2quat(p_FN)
#    
#    print 'p_FN', p_FN
#    print 'q_FN', quat_FN
#    
#    p_FN = ehatphi2mrp(ehat_FN, phi_FN)
#    ehat_FN, phi_FN = mrp2ehatphi(p_FN)
#    
#    print 'p_FN', p_FN
#    print 'ehat', ehat_FN
#    print 'phi', phi_FN*180./pi
#    
#    p_FN = gibbs2mrp(gibbs_FN)
#    gibbs_FN = mrp2gibbs(p_FN)
#    
#    print 'p_FN', p_FN
#    print 'gibbs_FN', gibbs_FN
#    
#    p_FN = dcm2mrp(DCM_FN)
#    print p_FN
#    DCM_FN = mrp2dcm(p_FN)
#    
#    print 'p_FN', p_FN
#    print 'DCM_FN', DCM_FN
#    
#    
#    return
#
#
#def test_compositions():
#    
#    
#    eulerB = [30., -45., 60.]
#    eulerF = [10., 25., -15.]
#    
#    sequence = [3,2,1]
#    
#    DCM_BN = euler_angles(sequence, eulerB[0]*pi/180., eulerB[1]*pi/180.,
#                          eulerB[2]*pi/180.)
#    
#    DCM_FN = euler_angles(sequence, eulerF[0]*pi/180., eulerF[1]*pi/180.,
#                          eulerF[2]*pi/180.)
#                          
#    DCM_BF = dcm_composition(DCM_BN, DCM_FN.T)
#    
#    quat_FN = dcm2quat(DCM_FN)
#    quat_BF = dcm2quat(DCM_BF)
#    
#    gibbs_FN = dcm2gibbs(DCM_FN)
#    gibbs_BF = dcm2gibbs(DCM_BF)
#    
#    mrp_FN = dcm2mrp(DCM_FN)
#    mrp_BF = dcm2mrp(DCM_BF)
#    
##    print quat_FN
##    print quat_BF
##    print np.linalg.norm(quat_FN)
##    print np.linalg.norm(quat_BF)
#    
#    quat_BN = quat_composition(quat_BF, quat_FN)
#    gibbs_BN = gibbs_composition(gibbs_BF, gibbs_FN)
#    mrp_BN = mrp_composition(mrp_BF, mrp_FN)
#    
#    quat_BN2 = dcm2quat(DCM_BN)
#    gibbs_BN2 = dcm2gibbs(DCM_BN)
#    mrp_BN2 = dcm2mrp(DCM_BN)
#    
#    print quat_BN
#    print quat_BN2
#    print gibbs_BN
#    print gibbs_BN2
#    print mrp_BN
#    print mrp_BN2
#    
#    quat_BN = np.reshape([1./sqrt(2.), 1./sqrt(2.), 0., 0.], (4,1))
#    quat_FB = np.reshape([-0.5*sqrt(sqrt(3.)/2. + 1), -sqrt(2)/(4.*sqrt(2. + sqrt(3.))), sqrt(2)/(4.*sqrt(2. + sqrt(3.))), 0.5*sqrt(sqrt(3.)/2. + 1)], (4,1))
##    
##    print np.linalg.norm(quat_BN)
##    print np.linalg.norm(quat_FB)  
#    
#    gibbs_BN = quat2gibbs(quat_BN)
#    gibbs_FB = quat2gibbs(quat_FB)
#    
#    mrp_BN = quat2mrp(quat_BN)
#    mrp_FB = quat2mrp(quat_FB)
#    
#    quat_FN = quat_composition(quat_FB, quat_BN)
#    gibbs_FN = gibbs_composition(gibbs_FB, gibbs_BN)
#    mrp_FN = mrp_composition(mrp_FB, mrp_BN)
#    
##    print quat_BN
##    print quat_FB
#    
#    print
#    print quat_FN
#    print gibbs_FN
#    
#    
#    DCM_FN = np.array([[0.5, sqrt(3)/2., 0.], [0., 0., 1.], [sqrt(3)/2., -0.5, 0.]])
#    
#    quat_FN = dcm2quat(DCM_FN)
#    gibbs_FN = dcm2gibbs(DCM_FN)
#    gibbs_FN2 = quat2gibbs(quat_FN)
#    mrp_FN2 = dcm2mrp(DCM_FN)
#    
#    print quat_FN
#    print gibbs_FN
#    print gibbs_FN2
#    
#    print
#    print mrp_FN
#    print mrp_FN2
#    
#    return
#    
#
#def test_rotations():
#    
#    eulerB = [30., -45., 60.]
#    
#    sequence = [3,2,1]
#    
#    DCM_BN = euler_angles(sequence, eulerB[0]*pi/180., eulerB[1]*pi/180.,
#                          eulerB[2]*pi/180.)
#                          
#    quat_BN = dcm2quat(DCM_BN)
#    quat_NB = quat_inverse(quat_BN)
#    
#    print quat_BN
#    print quat_NB
#    
#    vn = np.random.rand(3,1)
#    
#    vb = dcm_rotate(DCM_BN, vn)
#    vb2 = quat_rotate(quat_BN, vn)
#    
#    vn2 = quat_rotate(quat_NB, vb)
#    
#    
#    print 'vn', vn
#    print 'vb', vb
#    print 'vb2', vb2
#    
#    print
#    print 'vn2', vn2
#    
#    
#    return
#    
#
#def test_kinematics():
#    
#    
#    return
#    
#    
#def test_dynamics():
#    
#    return
#    
#    
##test_skew_matrix()
#
##test_parameter_conversions()
#
##test_compositions()
#
##test_rotations()
