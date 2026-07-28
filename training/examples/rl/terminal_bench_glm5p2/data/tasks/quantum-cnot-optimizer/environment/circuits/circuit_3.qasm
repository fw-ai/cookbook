OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];

h q[0];
h q[1];
h q[2];
h q[3];

cx q[0], q[1];
rz(0.5) q[0];
z q[0];
cx q[0], q[1];

cx q[0], q[2];
cx q[2], q[3];
cx q[1], q[3];

cx q[1], q[2];
x q[2];
h q[3];
cx q[1], q[2];

cx q[0], q[3];
s q[0];
t q[0];
cx q[0], q[3];
