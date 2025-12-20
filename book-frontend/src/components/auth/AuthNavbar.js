import React, { useState } from 'react';
import { useBetterAuth } from './BetterAuthProvider';

const AuthNavbar = () => {
  const { isAuthenticated, user, signout } = useBetterAuth();
  const [isOpen, setIsOpen] = useState(false);

  const toggleDropdown = (e) => {
    e.preventDefault();
    setIsOpen(!isOpen);
  };

  const closeDropdown = () => {
    setIsOpen(false);
  };

  if (isAuthenticated && user) {
    return (
      <div className={`navbar__item navbar__dropdown ${isOpen ? 'dropdown--show' : ''}`}>
        <div className="dropdown dropdown--right dropdown--navbar">
          <a
            href="#"
            className="navbar__link dropdown__link"
            role="button"
            aria-haspopup="true"
            aria-expanded={isOpen}
            onClick={toggleDropdown}
          >
            {user.name || user.email}
            <span className={`navbar__arrow ${isOpen ? 'navbar__arrow--up' : 'navbar__arrow--down'}`}></span>
          </a>
          {isOpen && (
            <ul className="dropdown__menu" onMouseLeave={closeDropdown}>
              <li>
                <a href="/auth/profile" className="dropdown__link" onClick={closeDropdown}>
                  Profile
                </a>
              </li>
              <li>
                <button
                  onClick={() => { signout(); closeDropdown(); }}
                  className="dropdown__link"
                  style={{ background: 'none', border: 'none', padding: '0', textAlign: 'left', width: '100%' }}
                >
                  Sign Out
                </button>
              </li>
            </ul>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="navbar__items navbar__items--right">
      <a href="/auth/signup" className="navbar__item navbar__link">
        Start Course
      </a>
      <a href="/auth/signin" className="navbar__item navbar__link">
        Sign In
      </a>
    </div>
  );
};

export default AuthNavbar;