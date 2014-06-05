%
% facepp class
%

classdef facepp
    properties (Hidden)
        key = [];
        secret = [];
        site = [];
    end
    methods
        function api = facepp(api_key, api_secret, api_region)
            % Get api_key, api_secret from your App on website.
            % Api_region can be CN(default) or US. Choose US when your API server
            % is Amazon
            api.key = api_key;
            api.secret = api_secret;
            if nargin < 3
                api_region = 'CN';
            end
            switch api_region
                case 'CN'
                    api.site = 'http://apicn.faceplusplus.com';
                case 'US'
                    api.site = 'http://apius.faceplusplus.com';
                otherwise
                    error('Not a valid region');
            end
        end
        
        function results = detect_file(api, file_path, attributes)
            % Detect faces from the image file, return attributes.
            % Attributes Can be all(default), none or a comma-separated list of desired attributes.
            % Currently supported attributes are: gender, age, race, smiling, glass and pose.
            % Example:
            %           detect_file(api, img, 'gender,age')
            fid = fopen(file_path, 'rb');
            data = fread(fid, Inf, '*uint8');
            fclose(fid);
            results = api.detect_buffer(data, attributes);
        end
        
        function results = detect_buffer(api, data, attributes)            
            str = urlreadpost(strcat(api.site, '/detection/detect'), ...
                { 'api_key', api.key, 'api_secret', api.secret, 'img', data, ...
                'attribute', attributes });
            results = parse_json(str);
        end
        
        function results = landmark(api, face_id, landmark_type)
            % Return the facial parts and contour locations in the detected faces
            % Type means number of key points, 83p and 25p are supported.
            % Example:
            %           landmark(api, face_id, '25p')
            
            str = urlreadpost(strcat(api.site, '/detection/landmark'), ...
                { 'api_key', api.key, 'api_secret', api.secret, 'face_id', face_id, ...
                'type', landmark_type });
            results = parse_json(str);
        end
    end
end