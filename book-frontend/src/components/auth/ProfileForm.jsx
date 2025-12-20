import React, { useState, useEffect } from 'react';
import { useBetterAuth } from './BetterAuthProvider';

const ProfileForm = ({ onProfileUpdateSuccess, onProfileUpdateError }) => {
  const { updateProfile, getProfile } = useBetterAuth();
  const [profileData, setProfileData] = useState({
    name: '',
    software_background: '',
    hardware_background: ''
  });
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [errors, setErrors] = useState({});

  const softwareBackgroundOptions = [
    'No Programming Experience',
    'Basic Python',
    'Intermediate Programming',
    'Advanced Programming'
  ];

  const hardwareBackgroundOptions = [
    'No Robotics Experience',
    'Basic Electronics',
    'Intermediate Robotics',
    'Advanced Robotics'
  ];

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      setLoading(true);
      const userData = await getProfile();
      setProfileData({
        name: userData.name || '',
        software_background: userData.software_background || '',
        hardware_background: userData.hardware_background || ''
      });
    } catch (err) {
      console.error('Error loading profile:', err);
    } finally {
      setLoading(false);
    }
  };

  const validateForm = () => {
    const newErrors = {};

    // Software background validation
    if (!profileData.software_background) {
      newErrors.software_background = 'Software background is required';
    }

    // Hardware background validation
    if (!profileData.hardware_background) {
      newErrors.hardware_background = 'Hardware background is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setUpdating(true);

    try {
      const updatedProfile = await updateProfile(profileData);

      if (onProfileUpdateSuccess) {
        onProfileUpdateSuccess(updatedProfile);
      }
    } catch (err) {
      if (onProfileUpdateError) {
        onProfileUpdateError(err.message || 'An error occurred while updating profile');
      }
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-md mx-auto bg-white p-8 rounded-lg shadow-md">
        <h2 className="text-2xl font-bold mb-6 text-center">Loading Profile...</h2>
        <div className="text-center">Please wait while we load your profile...</div>
      </div>
    );
  }

  return (
    <div className="max-w-md mx-auto bg-white p-8 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-center">Update Profile</h2>

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
            Name
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={profileData.name}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Your name"
          />
        </div>

        <div className="mb-4">
          <label htmlFor="software_background" className="block text-sm font-medium text-gray-700 mb-1">
            Software Background
          </label>
          <select
            id="software_background"
            name="software_background"
            value={profileData.software_background}
            onChange={handleChange}
            className={`w-full px-3 py-2 border ${errors.software_background ? 'border-red-500' : 'border-gray-300'} rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500`}
          >
            <option value="">Select your software background</option>
            {softwareBackgroundOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          {errors.software_background && <p className="mt-1 text-sm text-red-600">{errors.software_background}</p>}
          <p className="mt-1 text-xs text-gray-500">This helps us personalize your learning experience</p>
        </div>

        <div className="mb-6">
          <label htmlFor="hardware_background" className="block text-sm font-medium text-gray-700 mb-1">
            Hardware Background
          </label>
          <select
            id="hardware_background"
            name="hardware_background"
            value={profileData.hardware_background}
            onChange={handleChange}
            className={`w-full px-3 py-2 border ${errors.hardware_background ? 'border-red-500' : 'border-gray-300'} rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500`}
          >
            <option value="">Select your hardware background</option>
            {hardwareBackgroundOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          {errors.hardware_background && <p className="mt-1 text-sm text-red-600">{errors.hardware_background}</p>}
          <p className="mt-1 text-xs text-gray-500">This helps us recommend appropriate examples and exercises</p>
        </div>

        <button
          type="submit"
          disabled={updating}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {updating ? 'Updating Profile...' : 'Update Profile'}
        </button>
      </form>
    </div>
  );
};

export default ProfileForm;