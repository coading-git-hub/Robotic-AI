import React, { useState, useEffect } from 'react';

const SignupForm = ({ onSignup, onError }) => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    software_background: '',
    hardware_background: ''
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);
  const [isFocused, setIsFocused] = useState({});

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

  const validateForm = () => {
    const newErrors = {};

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length !== 8) {
      newErrors.password = 'Password must be exactly 8 characters';
    }

    if (!formData.software_background) {
      newErrors.software_background = 'Software background is required';
    }

    if (!formData.hardware_background) {
      newErrors.hardware_background = 'Hardware background is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
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

  const handleFocus = (name) => {
    setIsFocused(prev => ({
      ...prev,
      [name]: true
    }));
  };

  const handleBlur = (name) => {
    setIsFocused(prev => ({
      ...prev,
      [name]: false
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    try {
      // Prepare data for better-auth compatibility
      // The backend expects the custom fields but we'll send a compatible format
      const signupData = {
        email: formData.email,
        password: formData.password,
        name: formData.name,
        // Include background data if needed by backend
        software_background: formData.software_background,
        hardware_background: formData.hardware_background
      };

      const result = await onSignup(signupData);
      if (result) {
        // Reset form on successful signup
        setFormData({
          email: '',
          password: '',
          name: '',
          software_background: '',
          hardware_background: ''
        });
      }
    } catch (err) {
      onError && onError(err.message || 'An error occurred during signup');
    } finally {
      setLoading(false);
    }
  };

  const getPasswordStrength = (password) => {
    if (password.length === 0) return { level: 0, text: '', color: '' };
    if (password.length < 8) return { level: 1, text: 'Too short', color: 'bg-red-500' };
    if (password.length === 8) return { level: 2, text: 'Valid', color: 'bg-green-500' };
    return { level: 1, text: 'Too long', color: 'bg-red-500' };
  };

  const strength = getPasswordStrength(formData.password);

  return (
    <div className="auth-form-container">
      <div className="auth-card">
        <div className="text-center mb-8">
          <h2 className="auth-title">Create Account</h2>
          <p className="auth-subtitle">Join our community to get personalized learning experience</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form" autoComplete="off">
          <div className="auth-input-group">
            <div className="auth-input-wrapper">
              <div className="relative">
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  onFocus={() => handleFocus('name')}
                  onBlur={() => handleBlur('name')}
                  className={`auth-input ${
                    isFocused.name || formData.name ? 'auth-input-focused' : ''
                  }`}
                  placeholder=" "
                />
                <label
                  htmlFor="name"
                  className={`auth-input-label ${
                    isFocused.name || formData.name
                      ? 'auth-input-label-focused'
                      : 'auth-input-label-normal'
                  }`}
                >
                  Full Name
                </label>
                {(isFocused.name || formData.name) && (
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                    <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                )}
              </div>
              {errors.name && <p className="auth-error-message"><svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg> {errors.name}</p>}
            </div>

            <div className="auth-input-wrapper">
              <div className="relative">
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  onFocus={() => handleFocus('email')}
                  onBlur={() => handleBlur('email')}
                  className={`auth-input ${
                    isFocused.email || formData.email ? 'auth-input-focused' : ''
                  }`}
                  placeholder=" "
                />
                <label
                  htmlFor="email"
                  className={`auth-input-label ${
                    isFocused.email || formData.email
                      ? 'auth-input-label-focused'
                      : 'auth-input-label-normal'
                  }`}
                >
                  Email Address
                </label>
                {(isFocused.email || formData.email) && (
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                    <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                )}
              </div>
              {errors.email && <p className="auth-error-message"><svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg> {errors.email}</p>}
            </div>

            <div className="auth-input-wrapper">
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  onFocus={() => handleFocus('password')}
                  onBlur={() => handleBlur('password')}
                  maxLength={8}
                  className={`auth-input ${
                    isFocused.password || formData.password ? 'auth-input-focused' : ''
                  } pr-12`}
                  placeholder=" "
                />
                <label
                  htmlFor="password"
                  className={`auth-input-label ${
                    isFocused.password || formData.password
                      ? 'auth-input-label-focused'
                      : 'auth-input-label-normal'
                  }`}
                >
                  Password
                </label>
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-500 hover:text-gray-700 transition-colors duration-200"
                >
                  {showPassword ? (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  )}
                </button>
              </div>
              {errors.password && <p className="auth-error-message"><svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg> {errors.password}</p>}
              {!errors.password && (
                <p className="text-xs text-gray-500 mt-1">Password must be exactly 8 characters</p>
              )}

              {formData.password && (
                <div className="mt-2">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-600">Password strength</span>
                    <span className={`text-xs font-medium ${strength.color.replace('bg-', 'text-')}`}>{strength.text}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${strength.color}`}
                      style={{ width: strength.level === 2 ? '100%' : strength.level === 1 ? '50%' : '0%' }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="auth-input-wrapper">
                <label htmlFor="software_background" className="block text-sm font-medium text-gray-700 mb-2">
                  Software Background
                </label>
                <div className="relative">
                  <select
                    id="software_background"
                    name="software_background"
                    value={formData.software_background}
                    onChange={handleChange}
                    onFocus={() => handleFocus('software_background')}
                    onBlur={() => handleBlur('software_background')}
                    className={`w-full px-4 py-3 rounded-lg border-2 transition-all duration-200 focus:outline-none appearance-none ${
                      isFocused.software_background ? 'auth-input-focused' : 'border-gray-200'
                    } bg-white shadow-sm focus:shadow-md pr-10`}
                  >
                    <option value="">Select...</option>
                    {softwareBackgroundOptions.map(option => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                    </svg>
                  </div>
                </div>
                {errors.software_background && <p className="auth-error-message"><svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg> {errors.software_background}</p>}
              </div>

              <div className="auth-input-wrapper">
                <label htmlFor="hardware_background" className="block text-sm font-medium text-gray-700 mb-2">
                  Hardware Background
                </label>
                <div className="relative">
                  <select
                    id="hardware_background"
                    name="hardware_background"
                    value={formData.hardware_background}
                    onChange={handleChange}
                    onFocus={() => handleFocus('hardware_background')}
                    onBlur={() => handleBlur('hardware_background')}
                    className={`w-full px-4 py-3 rounded-lg border-2 transition-all duration-200 focus:outline-none appearance-none ${
                      isFocused.hardware_background ? 'auth-input-focused' : 'border-gray-200'
                    } bg-white shadow-sm focus:shadow-md pr-10`}
                  >
                    <option value="">Select...</option>
                    {hardwareBackgroundOptions.map(option => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                    </svg>
                  </div>
                </div>
                {errors.hardware_background && <p className="auth-error-message"><svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg> {errors.hardware_background}</p>}
              </div>
            </div>

            <div className="text-xs text-gray-500 bg-blue-50 p-3 rounded-lg">
              <p className="flex items-start">
                <svg className="w-4 h-4 mr-2 mt-0.5 text-blue-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Your background information helps us personalize content to your experience level.
              </p>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`auth-button ${
              loading
                ? 'auth-button-loading'
                : 'auth-button-active'
            }`}
          >
            <div className="flex items-center justify-center">
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creating Account...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
                  </svg>
                  Sign Up
                </>
              )}
            </div>
          </button>
        </form>
      </div>

      <div className="auth-footer">
        <p>Already have an account? <a href="/auth/signin" className="auth-link">Sign in here</a></p>
      </div>
    </div>
  );
};

export default SignupForm;