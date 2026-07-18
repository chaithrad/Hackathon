clc; clear;

 

%% PARAMETERS

fs = 100e6;

fop = 6e9;

 

num_samples = 1;

 

% receivers (fixed)

pos2 = [-300;0;0];

pos3 = [300;0;0];

pos4 = [-100;300;0] 

vel2 = [0;0;0];

 

%% STORAGE

X = cell(num_samples,1);   % store y5 (large matrices)

X_ts  = cell(num_samples,1);   % time signals

Y = zeros(num_samples,4);  % [x, y, velocity]

 

for i = 1:num_samples

 

    %% RANDOM TRANSMITTER POSITION

    pos1 = [0; 100; 0];%[rand*500+50; rand*500+50; 50];

 

    %% RANDOM VELOCITY

    vx = 50;%rand*200 - 100 + 50;
    vy = 10;%rand*200 - 100;

    vel1 = [vx;vy;0];

 

    %% SIGNAL GENERATION

    x1 = ((rand(1,1e6)-(1/2)) + 1j*(rand(1,1e6)-(1/2))).*2;

 

    % Optional filter (skip if not available)

    h = lowpass_100MHz;

    y1 = filter(h,1,x1);

   % y1 = x1;

 

    %% PROPAGATION

    sProp = phased.FreeSpace('SampleRate',fs,'OperatingFrequency',fop);

 

    y  = step(sProp,y1',pos1,pos2,vel1,vel2);

    y3 = step(sProp,y1',pos1,pos3,vel1,vel2);

    y4 = step(sProp,y1',pos1,pos4,vel1,vel2);

    %% SAVE TIME SIGNAL (MULTIMODAL)

    %%ts = [y'; y3'];   % 2-channel signal

    X_ts{i} = 1/fs;

 

    %% BLOCK PROCESSING

    A1 = y';

    A2 = y3';

    A3= y4';

 

    for k = 1:1000

        z1(k,1:1000) = A1((k-1)*1000 + 1: k*1000);

        z2(k,1:1000) = A2((k-1)*1000 + 1: k*1000);

        z4(k,1:1000) = A3((k-1)*1000 + 1: k*1000);

        z3(k,1:1000) = fft(z1(k,:)) .* conj(fft(z2(k,:)));
        z5(k,1:1000) = fft(z4(k,:)) .* conj(fft(z2(k,:)));

    end

 

    %% DELAY-DOPPLER MAP

    y4 = fft2(z3);

    y5 = abs(y4);

    y6 = fft2(z5);

    y7 = abs(y6);

    %% STORE

    X1{i} = y5;   % size 1000×1000
    X2{i} = y7;

    Y(i,:) = [pos1(1), pos1(2), vx, vy];

 

    fprintf("Sample %d done\n", i);

end

 

%% SAVE DATASET

save('matlab_dataset.mat','X1','X2','Y','X_ts','-v7.3');  % IMPORTANT for large data

 

disp("Dataset saved!");

 

 



% choose sample index

i = 1;   % or any sample



y5_sample = X1{i};

y7_sample = X2{i};

% find peak

[row1, col1] = find(y5_sample == max(y5_sample(:)));
[row2, col2] = find(y7_sample == max(y7_sample(:)));


figure;



%  Delay-Doppler Map

subplot(2,2,1);

imagesc(y5_sample);

colormap jet; colorbar;

hold on;

plot(col1, row1, 'ro', 'MarkerSize', 10, 'LineWidth', 2);

title('Delay-Doppler Map');

xlabel('Delay'); ylabel('Doppler');



%  Delay Profile

subplot(2,2,2);

plot(abs(y5_sample(row1,:)));

title('Delay Profile');

xlabel('Delay'); ylabel('Energy');



% Doppler Profile

subplot(2,2,3);

plot(abs(y5_sample(:,col1)));

title('Doppler Profile');

xlabel('Doppler'); ylabel('Energy');



%  3D surface (optional but useful)

subplot(2,2,4);

surf(y5_sample, 'EdgeColor','none');

title('3D View');

xlabel('Delay'); ylabel('Doppler'); zlabel('Magnitude');

view(45,45);

%%--------------------------------
figure;



%  Delay-Doppler Map

subplot(2,2,1);

imagesc(y7_sample);

colormap jet; colorbar;

hold on;

plot(col2, row2, 'ro', 'MarkerSize', 10, 'LineWidth', 2);

title('Delay-Doppler Map');

xlabel('Delay'); ylabel('Doppler');



%  Delay Profile

subplot(2,2,2);

plot(abs(y7_sample(row2,:)));

title('Delay Profile');

xlabel('Delay'); ylabel('Energy');



% Doppler Profile

subplot(2,2,3);

plot(abs(y7_sample(:,col2)));

title('Doppler Profile');

xlabel('Doppler'); ylabel('Energy');



%  3D surface (optional but useful)

subplot(2,2,4);

surf(y7_sample, 'EdgeColor','none');

title('3D View');

xlabel('Delay'); ylabel('Doppler'); zlabel('Magnitude');

view(45,45);

% subplot(2,2,1);
% 
% imagesc(y5); colormap jet; colorbar;
% 
% hold on;
% 
% plot(col,row,'ro','MarkerSize',10,'LineWidth',2);
% 
% title('Delay-Doppler Map');
% 
% xlabel('Range'); ylabel('Velocity');
% 
% 
% 
% subplot(2,2,2);
% 
% plot(real(ts(1,1:2000)));
% 
% title('Time Signal');
% 
% xlabel('Samples'); ylabel('Amplitude');
% 
% 
% 
% subplot(2,2,3);
% 
% plot(abs(y5(row,:)));
% 
% title('Range Profile');
% 
% xlabel('Range'); ylabel('Energy');
% 
% 
% 
% subplot(2,2,4);
% 
% plot(abs(y5(:,col)));
% 
% title('Velocity Profile');
% 
% xlabel('Velocity'); ylabel('Energy');