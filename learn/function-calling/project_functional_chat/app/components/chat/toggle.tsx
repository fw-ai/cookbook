import { useState } from 'react';

export default function Toggle(props: {
  showText: string,
  hideText: string,
  children: React.ReactNode;
}) {
  const [isVisible, setIsVisible] = useState(false);

  const toggleVisibility = () => {
    setIsVisible(!isVisible);
  };

  return (
    <div>
      <div className="flex justify-end">
        <a href="#" onClick={toggleVisibility} className="text-sky-600 hover:text-sky-700 flex items-center text-xs">
          {isVisible ? props.hideText : props.showText}
        </a>
      </div>
      {isVisible && <div className="mt-4">
        {props.children}
      </div>}
    </div>
  );
}
