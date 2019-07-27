%Jon M. Ernstberger
%verson 2.0
%7/13/2006
%
%This is a standard Quasi-Newton Solver
%
%Inputs:
%
%	1.  q0-Initial parameter estimate to be use in function string passed in call
%	2.  Fcn-objective functional mentioned in 1.  Functional name must be passed as a string or with handle.
%  	3.  max_iter-Maximum number of QN iterations
%  	4.  q_tol-Tolerance on parameter for sufficient convergence
%  	5.  f_tol-Toler on function for sufficient functional convergence.
%	6.  Hess_choice-option for choice of Hessian matrix.  'BFGS' implements the Hessian approximate
%		while anything else forces a finite-difference approximate
%  	
%  Outputs:
%  	1.  q_opt-Optimized parameter to be estimated
%  	2.  J_opt-Value of objective functional at optimal parameter.
%
%Method:
%  	Standard Quadi-Newton used for parameter estimation:
%  	
%  		q_{k+1}=q_{k}-Hess(J(q_k))^{-1}Grad(J(q_k))
%  		
%  	where q_k is the parameter estimate at iterate k, J(q_k) is the value of the 
%  	objective functional at q_k.  Hess(J(q_k)) is the Hessian matrix of the objective functional
%  	at q_k and Grad(J(q_k)) is the gradient of the objective functional at q_k.
%Fcn,q0,max_iter,q_tol,f_tol,Hess_choice)

function [varargout]=quasi_newton(varargin)

	%Set defaults and initialize
	if nargin==1 & isstruct(varargin{1})
		input=varargin{1};
	
		if ~isfield(input,'fcn') 
			error('Must specify an objective functional');
		else 
			Fcn=input.fcn;
		end;

		if ~isfield(input,'initial')
			error('Must include an initial iterate');
		else 
			q0=input.initial;
		end;

		if ~isfield(input,'max_iter') 
			max_iter=25*length(q0);
		else 
			max_iter=input.max_iter;
		end;
		
		if ~isfield(input,'q_tol') 
			q_tol=1e-6;
		else 
			q_tol=input.q_tol;
		end;
		
		if ~isfield(input,'f_tol') 
			f_tol=1e-6;
		else 
			f_tol=input.f_tol;
		end;
		
		if ~isfield(input,'Hess_choice') 
			Hess_choice='BFGS';
		else 
			Hess_choice=input.Hess_choice;
		end;

	elseif  nargin>1

		if nargin<2 & ~isstruct(varargin{1})
			error('Not enough inputs');
		elseif nargin==2
			Fcn=varargin{1};
			q0=varargin{2};
			max_iter=25*length(q0);	
			q_tol=1e-6;
			f_tol=1e-6;
			Hess_choice='BFGS';
		elseif nargin==3
			Fcn=varargin{1};
			q0=varargin{2};
			max_iter=varargin{3};
			q_tol=1e-6;
			f_tol=1e-6;
			Hess_choice='BFGS';
		elseif nargin==4
			Fcn=varargin{1};
			q0=varargin{2};
			max_iter=varargin{3};
			q_tol=varargin{4};
			f_tol=1e-6;
			Hess_choice='BFGS';
		elseif nargin==5
			Fcn=varargin{1};
			q0=varargin{2};
			max_iter=varargin{3};
			q_tol=varargin{4};
			f_tol=varargin{5};
			Hess_choice='BFGS';
		elseif nargin==6
			Fcn=varargin{1};
			q0=varargin{2};
			max_iter=varargin{3};
			q_tol=varargin{4};
			f_tol=varargin{5};
			Hess_choice=varargin{6};
		else
			s=['Discarding extra inputs...'];
			disp(s);
		end;
	end;
	
	
	%storage of iteration
	q_hist(:,1)=q0(:);
	J_hist(:,1)=feval(Fcn,q_hist(:,1));
	
	i=1;
	fcncnt=0;
	flag=0;

	%Choosing Default Hessian Matrix
	if strcmp(Hess_choice,'BFGS');
		H=eye(length(q0));
	else
		H=hess_fwd(Fcn,q_hist(:,1));
	end;
	g=grad(Fcn,q_hist(:,i));fcncnt=fcncnt+length(q0)+1;
	
	while i<=max_iter & flag==0
		
		%Used for displaying iteration count, etc.
		if mod(i-1,20)==0
			printf('\nIterations\t Function Evaluations \t Cost\n');
		end;
		
		
		%Newton Direction
		%if cond(H)<=eps
		%	[m,n]=size(H);
		%	Hinv=zeros(m,n);
		%	for z=1:m
		%		Hinv(z,z)=1/H(z,z);
		%	end;

		%	H=Hinv*H;
		%	g=Hinv*g;
		%end;
		s=H\g;
		
		%Line Search Parameter
		y=1;
		LSok=1;countLS=1;
		%Newton Step w. Line Search
		while LSok & countLS<=5
			q_hist(:,i+1)=q_hist(:,i)-y*s;
			
			if norm(feval(Fcn,q_hist(:,i+1)))>norm(feval(Fcn,q_hist(:,i)))
				y=y/2;
				countLS=countLS+1;
			else
				LSok=0;
			end;
			fcncnt=fcncnt+2;
		end
		
		J_hist(i+1)=feval(Fcn,q_hist(:,i+1));fcncnt=fcncnt+1;
		
		%check for q_tol
		val=norm(q_hist(:,i+1)-q_hist(:,i));
		if val<q_tol
			flag=1;
			disp('Tolerance on parameter met.');
		end;
		
		%check for f_tol
		val=norm(feval(Fcn,q_hist(:,i+1))-feval(Fcn,q_hist(:,i)));fcncnt=fcncnt+2;
		if val<f_tol
			flag=2;
			disp('Tolerance on functional convergence met.');
		end;
		
		%output values to screen
		%s=[num2str(i),'  ',num2str(J_hist(end))];
		%disp(s);
		printf('%g\t\t %g\t\t\t %e\n',i,fcncnt,J_hist(end));
		
		%Compute new gradient
		g1=grad(Fcn,q_hist(:,i+1));fcncnt=fcncnt+length(q0)+1;
		
		%Update Hessian
		if strcmp(Hess_choice,'BFGS')
			s=q_hist(:,i+1)-q_hist(:,i);
			y=g1-g;
			
			ip1=y'*s;
			ip2=H*s;
			
			if ip1<=0
				H=eye(length(q_hist(:,i+1)));
			end;
			
			H=H-((ip2)'*s)\(ip2*ip2')+(ip1)\(y*y');
		else
			H=hess_fwd(Fcn,q_hist(:,i+1));fcncnt=fcncnt+2*length(q0)+2;
		end;

		
		%Update Gradient and iteration counter
		g=g1;
		i=i+1;
	end;

	index=find(J_hist==min(J_hist));

	if nargout==1
		varargout{1}=q_hist(:,index);
	elseif nargout==2
		varargout{1}=q_hist(:,index);
		varargout{2}=min(J_hist);
	elseif nargout==3
		varargout{1}=q_hist(:,index);
		varargout{2}=min(J_hist);
		varargout{3}=q_hist;
	elseif nargout==4
		varargout{1}=q_hist(:,index);
		varargout{2}=min(J_hist);
		varargout{3}=q_hist;
		varargout{4}=J_hist;
	end;
return;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Forward difference gradient approximation.                                 %
%                                                                           %
%Input:                                                                     %
%  	1.  obj_fcn-Objective functional                                    %
%  	2.  q-Initial parameter iterate.                                    %
%  	                                                                    %
%  Output:                                                                  %
%  	1.  Grad-gradient vector computed using a forward difference        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
function Grad=grad(obj_fcn,q)

	new_q=q;
	
	J1=feval(obj_fcn,q);
	
	h=eps^(1/3)*q;
	
	for i=1:length(q)
		
		new_q(i)=new_q(i)+h(i);
		J2(i)=feval(obj_fcn,new_q);
	
		new_q=q;
	end;
	
	Grad=(J2-J1)./h';
	Grad=Grad(:);

return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Forward difference Hessian Matrix approximation                            %
%                                                                           %
%Input:                                                                     %
%  	1.  obj_fcn-Objective functional                                    %
%  	2.  q-Initial parameter iterate.                                    %
%  	                                                                    %
%  Output:                                                                  %
%  	1.  Hess-Hessian matrix computed using a forward difference         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Hess=hess_fwd(obj_fcn,q)

	h=eps^(1/3)*norm(q);
	
	new_q=q;
	grad1=grad(obj_fcn,q);
	for i=1:length(q)
		new_q(i)=new_q(i)+h;
		
		grad2=grad(obj_fcn,new_q);
		Hess(:,i)=(grad2-grad1)/h;
		
		new_q=q;
	end;

return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
