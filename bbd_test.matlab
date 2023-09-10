%%%%%%%%%%%%%MATLAB Solver Full%%%%%%%%%%%%%
A_d =  A_org(28:50, 28:50);  %done

A0=zeros(36, 36);
A0_a = A_org(1:13, 1:13);  %A
A0_c = A_org(28:50, 1:13); %C
A0_b = A_org(1:13, 28:50); %B

A0(1:13, 1:13) = A0_a;  %A
A0(14:36, 1:13) = A0_c; %C
A0(1:13, 14:36) = A0_b; %B
A0(14:36, 14:36) = A_d; %D



A1=zeros(37, 37);   
A1_a = A_org(14:27, 14:27); %A
A1_c= A_org(28:50, 14:27);  %C
A1_b = A_org(14:27, 28:50); %B

A1(1:14, 1:14) = A1_a;   %A
A1(15:37, 1:14) = A1_c;  %C
A1(1:14, 15:37) = A1_b;  %B
A1(15:37, 15:37) = A_d;  %D


L0_mat = chol(A0_a, 'lower');
U0_mat = transpose(L0_mat);
CUinv0_mat = A0_c * inv(U0_mat);
LinvB0_mat = transpose(CUinv0_mat);

L1_mat = chol(A1_a, 'lower');
U1_mat = transpose(L1_mat);
CUinv1_mat = A1_c * inv(U1_mat);
LinvB1_mat = transpose(CUinv1_mat);

schur = A_d - (CUinv0_mat * LinvB0_mat + CUinv1_mat * LinvB1_mat);
schur_b = ones(23, 1) - (CUinv0_mat *(L0_mat\ones(13, 1)) + CUinv1_mat * (L1_mat\ones(14, 1)));

xc = inv(schur) * schur_b 
x0 = U0_mat\(L0_mat\(ones(13, 1) - A0_b * xc));   % LU x =y
x1 = U1_mat\(L1_mat\(ones(14, 1) - A1_b * xc));



%%%%%%%%%%%%%MATLAB Solver Lower Part%%%%%%%%%%%%%
A_d =  A_org(28:50, 28:50);  % done

A0 = zeros(36, 36);
A0_a = A_org(1:13, 1:13);  %A
A0_c = A_org(28:50, 1:13); %C
A0_b = transpose(A0_c);  %B

A0(1:13, 1:13) = A0_a;  %A
A0(14:36, 1:13) = A0_c; %C
% A0(1:13, 14:36) = A0_b; %B
A0(14:36, 14:36) = A_d; %D



A1=zeros(37, 37);   
A1_a = A_org(14:27, 14:27); %A
A1_c= A_org(28:50, 14:27);  %C
A1_b = A_org(14:27, 28:50); %B

A1(1:14, 1:14) = A1_a;   %A
A1(15:37, 1:14) = A1_c;  %C
% A1(1:14, 15:37) = A1_b;%B
A1(15:37, 15:37) = A_d;  %D


L0_mat_lower = chol(A0_a, 'lower');
U0_mat_lower = transpose(L0_mat_lower);
CUinv0_mat_lower = A0_c * inv(U0_mat_lower);
LinvB0_mat_lower = transpose(CUinv0_mat_lower);

L1_mat_lower = chol(A1_a, 'lower');
U1_mat_lower = transpose(L1_mat_lower);
CUinv1_mat_lower = A1_c * inv(U1_mat_lower);
LinvB1_mat_lower = transpose(CUinv1_mat_lower);


A_d_full = A_d + transpose(A_d) - diag(diag(A_d));

schur = A_d_full - (CUinv0_mat * LinvB0_mat + CUinv1_mat * LinvB1_mat);
schur_b = ones(23, 1) - (CUinv0_mat *(L0_mat\ones(13, 1)) + CUinv1_mat * (L1_mat\ones(14, 1)));

xc = inv(schur) * schur_b 
x0 = U0_mat\(L0_mat\(ones(13, 1) - A0_b * xc));   % LU x =y
x1 = U1_mat\(L1_mat\(ones(14, 1) - A1_b * xc));


