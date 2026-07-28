OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];

h q[0];
h q[1];
h q[2];
h q[3];
h q[4];

cx q[0], q[1];
rz(0.1) q[4];
cx q[1], q[2];
t q[4];
cx q[2], q[3];
s q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[2], q[3];
rz(0.2) q[0];
cx q[1], q[2];
cx q[0], q[1];

cx q[0], q[3];
cx q[2], q[4];
