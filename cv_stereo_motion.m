%===================================================
% Computer Vision: Stereo
% Allen Lau 
%===================================================

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 5.1 Fundamental Matrix %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Point Correspondences
% Read in two images 
imgl = imread('images/pic410.png');
imgr = imread('images/pic430.png');

% display image pair side by side
[ROWS,COLS,CHANNELS] = size(imgl);
disimg = [imgl imgr];
image(disimg);

% Total Number of control points
Nc = 8;
% Total Number of test points
Nt = 2;

%build pr, pl arrays with user input and plot 
cnt = 1;
hold;
while(cnt <= Nc+Nt)

% size of the rectangle to indicate point locations
dR = 50;
dC = 50;

% pick up a point in the left image and display it with a rectangle....
%%% if you loaded the point matches, comment the point picking up (3 lines)%%%
[X, Y] = ginput(1);
Cl = X(1); Rl = Y(1);
pl(cnt,:) = [Cl Rl 1];

% and draw it 
Cl= pl(cnt,1);  Rl=pl(cnt,2); 
rectangle('Curvature', [0 0], 'Position', [Cl Rl dC dR]);

% and then pick up the correspondence in the right image
%%% if you loaded the point matches, comment the point picking up (three lines)%%%

[X, Y] = ginput(1);
Cr = X(1); Rr = Y(1);
pr(cnt,:) = [Cr-COLS Rr 1];

% draw it
Cr=pr(cnt,1)+COLS; Rr=pr(cnt,2);
rectangle('Curvature', [0 0], 'Position', [Cr Rr dC dR]);
%plot(Cr+COLS,Rr,'r*');
drawnow;

cnt = cnt+1;
end

%%
%loading pl, pr files
load ./mat_files/pl.mat;
load ./mat_files/pr.mat;

%normalization of the correspondence point image locations 
%%normalization for the left image points
x_pl_mean = mean(pl(1:8,1));
y_pl_mean = mean(pl(1:8,2));
hyp_pl = sqrt(x_pl_mean^2 + y_pl_mean^2);
scale_pl = sqrt(2)/hyp_pl;
T_pl = scale_pl*[1, 0, -x_pl_mean;0,1,-y_pl_mean;0,0,1/scale_pl];

%normalization for the right image points 
x_pr_mean = mean(pr(1:8,1));
y_pr_mean = mean(pr(1:8,2));
hyp_pr = sqrt(x_pr_mean^2 + y_pr_mean^2);
scale_pr = sqrt(2)/hyp_pr;
T_pr = scale_pr*[1, 0, -x_pr_mean;0,1,-y_pr_mean;0,0,1/scale_pr];

%normalize the points 
for i = 1:1:8
    pl_norm(i,:) = T_pl * pl(i,:)';
    pr_norm(i,:) = T_pr * pr(i,:)';
end

%%%Eight-Point Algorithm to estimate F matrix%%%

%constructing the homogenous system 
for i = 1:1:8
    A(i,:) = [pl_norm(i,1)*pr_norm(i,1), pl_norm(i,1)*pr_norm(i,2), pl_norm(i,1), pl_norm(i,2)*pr_norm(1), pl_norm(i,2)*pr_norm(i,2), pl_norm(i,2), pr_norm(i,1), pr_norm(i,2),1];
end

% compute f_hat
[U,D,V_T] = svd(A);
x = V_T(9,:);
x = reshape(x,3,3);
[Ux,Dx,V_Tx]=svd(x);
Dx(3,3) = 0; 

%f_norm
F_norm = Ux*Dx*V_Tx; 

%denormalize F 
F=T_pr'*F_norm*T_pl; 

% epipoles 
[Ue,De,V_Te] = svd(F);
el = V_Te(:,3);
er = Ue(:,3);

%reading in data for testing 
% Read in two images 
imgl = imread('images/pic410.png');
imgr = imread('images/pic430.png');
% display image pair side by side
[ROWS,COLS,CHANNELS] = size(imgl);
disimg = [imgl imgr];
% Total Number of control points
Nc = 8;
% Total Number of test points
Nt = 2;

%accuracy 
image(disimg);
hold;
for cnt=1:1:Nc+Nt
  %determining the epipolar line parameters 
  an = F*pl(cnt,:)';
  %epipolar line on the right image 
  x = 0:COLS; 
  y = -(an(1)*x+an(3))/an(2);
  x = x+COLS;
  %plotting the left point and the right point 
  plot(pl(cnt,1),pl(cnt,2),'r*','LineWidth', 2);
  plot(pr(cnt,1)+COLS,pr(cnt,2),'b*','LineWidth', 2);
  %plotting the epipolar line 
  line(x,y,'Color', 'r','LineWidth', 1.5);
  %[X, Y] = ginput(1); %% the location doesn't matter, press mouse to continue...
  plot(pl(cnt,1),pl(cnt,2),'r*','LineWidth', 2);
  line(x,y,'Color', 'r','LineWidth', 1.5);
  %computing the error as the euclidean distance between the true location 
  %of the right correspondence point and the epipolar line 
  error(cnt) = abs(an(1)*(pr(cnt,1)) + an(2)*pr(cnt,2) + an(3))/sqrt(an(1)^2 + an(2)^2);
end 
hold off; 

%display error values 
disp('Error of Test Points:')
disp(error(9:10))

%%
%%%Automatically Finding the Correspondence Point%%%
image(disimg);
hold;
%automatically finding the correspondence point
window_side = 10;
for cnt=1:1:Nc+Nt
    an = F*pl(cnt,:)';
    %epipolar line on the right image 
    x = 0:COLS; 
    y = -(an(1)*x+an(3))/an(2);
    x = x+COLS;
    %plotting the left reference point and the epipolar line
    plot(pl(cnt,1),pl(cnt,2),'r*','LineWidth', 2);
    line(x,y,'Color', 'r','LineWidth', 1.5);
    %extract window around pl point
    l_window = disimg(round(pl(cnt, 1))-window_side:round(pl(cnt, 1))+window_side, round(pl(cnt, 2))-window_side:round(pl(cnt, 2))+window_side);
    %define arrays to track the correlations and indices 
    correlations = [];
    indices_x = [];
    indices_y = [];
    for i = 1:1:length(x)
        %make sure we are not out of bounds for the image
         if (round(x(i))-window_side>COLS && round(y(i))-window_side>0) && (round(y(i))+window_side<ROWS && round(x(i))+window_side<(2*COLS))
            %define the right window based on the epipolar line and the
            %defined window size 
            r_window = disimg(round(y(i))-window_side:round(y(i))+window_side, round(x(i))-window_side:round(x(i))+window_side);
            %compute the cross correlation between the left window and
            %right window 
            cross_correlation = xcorr2(l_window,r_window);
            %track our results in arrays 
            correlations(end+1) = max(cross_correlation(:));
            indices_x(end+1) = x(i);
            indices_y(end+1) = y(i);
        end 
    end
    %find the index of max cross correlation 
    index = find(correlations == max(correlations)); 
    corr_x = indices_x(index); 
    corr_y = indices_y(index);
    %draw a rectangle over the max cross correlation 
    rectangle('Position', [corr_x, corr_y, window_side*2, window_side*2], 'EdgeColor', 'b', 'LineWidth', 2);
end

hold off; 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 5.2 Feature-Based Matching %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%apply sobel kernel to the left and right images 
%sobel vertical gradient 
s1 = [[-1,0,1];[-2,0,2];[-1,0,1]];
s2 = [[-1,-2,-1];[0,0,0];[1,2,1]];

%defining images
imgl = imread('images/pic410.png');
imgr = imread('images/pic430.png');
%grayscale the images 
colormap("gray");
imgl_gray = uint8(round(sum(imgl,3)/3));
imgr_gray = uint8(round(sum(imgr,3)/3));
%apply vertical/horizontal sobel kernels on images
imgl_v = conv2(imgl_gray,s1);
imgl_h = conv2(imgl_gray,s2);
imgr_v = conv2(imgr_gray,s1);
imgr_h = conv2(imgr_gray,s2);
%combined gradient images 
imgl_edges = sqrt(imgl_v.^2 + imgl_h.^2);
imgr_edges = sqrt(imgr_v.^2 + imgr_h.^2);
%displaying 
disimg_edges = [imgl_edges imgr_edges];
image(disimg_edges);

% Total Number of control points
Nc = 8;
% Total Number of test points
Nt = 2;

%build pr, pl arrays with user input and plot 
cnt = 1;
hold;
while(cnt <= Nc+Nt)

% size of the rectangle to indicate point locations
dR = 50;
dC = 50;

% pick up a point in the left image and display it with a rectangle....
%%% if you loaded the point matches, comment the point picking up (3 lines)%%%
[X, Y] = ginput(1);
Cl = X(1); Rl = Y(1);
pl(cnt,:) = [Cl Rl 1];

% and draw it 
Cl= pl(cnt,1);  Rl=pl(cnt,2); 
rectangle('Curvature', [0 0], 'Position', [Cl Rl dC dR],'EdgeColor', 'red', 'LineWidth', 2);
drawnow;

cnt = cnt+1;
end

%Automatically Finding the Correspondence Point
%automatically finding the correspondence point
window_side = 20;
for cnt=1:1:Nc+Nt
    an = F*pl(cnt,:)';
    %epipolar line on the right image 
    x = 0:COLS; 
    y = -(an(1)*x+an(3))/an(2);
    x = x+COLS;
    %plotting the left reference point and the epipolar line
    plot(pl(cnt,1),pl(cnt,2),'r*','LineWidth', 2);
    line(x,y,'Color', 'r','LineWidth', 1.5);
    %extract window around pl point
    l_window = disimg_edges(round(pl(cnt, 1))-window_side:round(pl(cnt, 1))+window_side, round(pl(cnt, 2))-window_side:round(pl(cnt, 2))+window_side);
    %define arrays to track the correlations and indices 
    correlations = [];
    indices_x = [];
    indices_y = [];
    for i = 1:1:length(x)
        %make sure we are not out of bounds for the image
        if (round(x(i))-window_side>COLS && round(y(i))-window_side>0) && (round(y(i))+window_side<ROWS && round(x(i))+window_side<(2*COLS))
            %define the right window based on the epipolar line and the
            %defined window size 
            r_window = disimg_edges(round(y(i))-window_side:round(y(i))+window_side, round(x(i))-window_side:round(x(i))+window_side);
            %compute the cross correlation between the left window and
            %right window 
            cross_correlation = xcorr2(l_window,r_window);
            %track our results in arrays 
            correlations(end+1) = max(cross_correlation(:));
            indices_x(end+1) = x(i);
            indices_y(end+1) = y(i);
        end 
    end
    %find the index of max cross correlation 
    index = find(correlations == max(correlations)); 
    corr_x = indices_x(index); 
    corr_y = indices_y(index);
    %draw a rectangle over the max cross correlation 
    rectangle('Position', [corr_x, corr_y, window_side*2, window_side*2], 'EdgeColor', 'cyan', 'LineWidth', 2);
end

hold off; 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 5.3 Results on Points w/ Different Properties %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Point Correspondences
% Read in two images 
imgl = imread('images/pic410.png');
imgr = imread('images/pic430.png');

% display image pair side by side
[ROWS,COLS,CHANNELS] = size(imgl);
disimg = [imgl imgr];
image(disimg);

% Total Number of control points
Nc = 8;
% Total Number of test points
Nt = 2;

%build pr, pl arrays with user input and plot 
cnt = 1;
hold;
while(cnt <= Nc+Nt)

% size of the rectangle to indicate point locations
dR = 50;
dC = 50;

% pick up a point in the left image and display it with a rectangle....
%%% if you loaded the point matches, comment the point picking up (3 lines)%%%
[X, Y] = ginput(1);
Cl = X(1); Rl = Y(1);
pl(cnt,:) = [Cl Rl 1];

% and draw it 
Cl= pl(cnt,1);  Rl=pl(cnt,2); 
rectangle('Curvature', [0 0], 'Position', [Cl Rl dC dR]);

% and then pick up the correspondence in the right image
%%% if you loaded the point matches, comment the point picking up (three lines)%%%

[X, Y] = ginput(1);
Cr = X(1); Rr = Y(1);
pr(cnt,:) = [Cr-COLS Rr 1];

% draw it
Cr=pr(cnt,1)+COLS; Rr=pr(cnt,2);
rectangle('Curvature', [0 0], 'Position', [Cr Rr dC dR]);
%plot(Cr+COLS,Rr,'r*');
drawnow;

cnt = cnt+1;
end

%normalization of the correspondence point image locations 
%%normalization for the left image points
x_pl_mean = mean(pl(1:8,1));
y_pl_mean = mean(pl(1:8,2));
hyp_pl = sqrt(x_pl_mean^2 + y_pl_mean^2);
scale_pl = sqrt(2)/hyp_pl;
T_pl = scale_pl*[1, 0, -x_pl_mean;0,1,-y_pl_mean;0,0,1/scale_pl];

%normalization for the right image points 
x_pr_mean = mean(pr(1:8,1));
y_pr_mean = mean(pr(1:8,2));
hyp_pr = sqrt(x_pr_mean^2 + y_pr_mean^2);
scale_pr = sqrt(2)/hyp_pr;
T_pr = scale_pr*[1, 0, -x_pr_mean;0,1,-y_pr_mean;0,0,1/scale_pr];

%normalize the points 
for i = 1:1:8
    pl_norm(i,:) = T_pl * pl(i,:)';
    pr_norm(i,:) = T_pr * pr(i,:)';
end

%%%Eight-Point Algorithm to estimate F matrix%%%

%constructing the homogenous system 
for i = 1:1:8
    A(i,:) = [pl_norm(i,1)*pr_norm(i,1), pr_norm(i,1)*pl_norm(i,2), pr_norm(i,1), pr_norm(i,2)*pl_norm(i,1), pr_norm(i,2)*pl_norm(i,2), pl_norm(i,1), pl_norm(i,2), pr_norm(i,2),1];
end

% compute f_hat
[U,D,V_T] = svd(A);
x = V_T(9,:);
x = reshape(x,3,3);
[Ux,Dx,V_Tx]=svd(x);
Dx(3,3) = 0; 

%f_norm
F_norm = Ux*Dx*V_Tx; 

%denormalize F 
F=T_pr'*F_norm*T_pl; 

% epipoles 
[Ue,De,V_Te] = svd(F);
el = V_Te(:,3);
er = Ue(:,3);

%reading in data for testing 
% Read in two images 
imgl = imread('images/pic410.png');
imgr = imread('images/pic430.png');
% display image pair side by side
[ROWS,COLS,CHANNELS] = size(imgl);
disimg = [imgl imgr];
% Total Number of control points
Nc = 8;
% Total Number of test points
Nt = 2;

%accuracy 
% image(disimg);
% hold;
for cnt=1:1:Nc+Nt
  %determining the epipolar line parameters 
  an = F*pl(cnt,:)';
  %epipolar line on the right image 
  x = 0:COLS; 
  y = -(an(1)*x+an(3))/an(2);
  x = x+COLS;
  %plotting the left point and the right point 
  plot(pl(cnt,1),pl(cnt,2),'r*','LineWidth', 2);
  plot(pr(cnt,1)+COLS,pr(cnt,2),'b*','LineWidth', 2);
  %plotting the epipolar line 
  line(x,y,'Color', 'r','LineWidth', 1.5);
  %[X, Y] = ginput(1); %% the location doesn't matter, press mouse to continue...
  plot(pl(cnt,1),pl(cnt,2),'r*','LineWidth', 2);
  line(x,y,'Color', 'r','LineWidth', 1.5);
  %computing the error as the euclidean distance between the true location 
  %of the right correspondence point and the epipolar line 
  error(cnt) = abs(an(1)*(pr(cnt,1)) + an(2)*pr(cnt,2) + an(3))/sqrt(an(1)^2 + an(2)^2);
end 
hold off; 

%display error values 
disp('Error of Test Points:')
disp(error(9:10))

