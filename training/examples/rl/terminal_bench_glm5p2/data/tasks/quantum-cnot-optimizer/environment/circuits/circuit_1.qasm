OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];

h q[0];
h q[1];
h q[2];

cx q[0], q[1];
rz(0.3) q[0];
cx q[0], q[1];

cx q[1], q[2];
t q[1];
s q[1];
cx q[1], q[2];

cx q[0], q[2];
cx q[1], q[0];
